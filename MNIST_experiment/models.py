import tensorflow as tf 
import numpy as np 
import pathlib

# init_weights_path = pathlib.Path.home()/'Documents/gradschool/672/project/regularization_project/MNIST_experiment/init_weights.npy'
init_weights_path = '/home/brennan/672/regularization_project/MNIST_experiment/init_weights.npy'
# init_weights_path = '/content/MNIST_experiment/init_weights.npy'

class Baseline():
    def __init__(self, config):
        self.optimizer = tf.train.GradientDescentOptimizer(1e-2)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.is_training = tf.placeholder_with_default(True, shape=())

        self.batch_size = config['batch_size']

        self.mlp = self.get_mlp()
        self.build_graph()

    def get_mlp(self):
        weights = np.load(str(init_weights_path))
        return tf.keras.layers.Dense(10, activation='softmax', \
        kernel_initializer=tf.constant_initializer(weights))

    def build_datapipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data))\
            .batch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                  dataset.output_shapes)
        self.dset_init = iterator.make_initializer(dataset)
        return iterator

    def model(self, x):
        x = self.mlp(x)
        return x 
        
    def build_graph(self):
        self.x_data = tf.placeholder(np.float32, [None, 784])
        self.y_data = tf.placeholder(np.float32, [None, 10])

        iterator = self.build_datapipeline()
        xb, yb = iterator.get_next()

        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        # add for regularization layers (L1, L2, etc. subclasses)
        self.loss += tf.math.reduce_sum(self.mlp.losses)

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.w_grad = tf.gradients(self.loss, tf.compat.v1.trainable_variables())[0]
        self.w = tf.compat.v1.trainable_variables()[0]

        self.train_op = self.optimizer.minimize(self.loss)

class DropoutReg(Baseline):
    def __init__(self, config): 
        self.dropout_constant = config['dropout_constant']
        self.dropout = tf.keras.layers.Dropout(self.dropout_constant)
        super().__init__(config)

    def model(self, x):
        x = super().model(x)
        x = self.dropout(x, training=self.is_training)
        return x 

class SpectralReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        self.v = tf.random.normal((10, 1), mean=0., stddev=1.)
        super().__init__(config)

    def build_graph(self):
        self.x_data = tf.placeholder(np.float32, [None, 784])
        self.y_data = tf.placeholder(np.float32, [None, 10])

        iterator = self.build_datapipeline()
        xb, yb = iterator.get_next()

        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        # apply spectral norm reg. 
        grads = tf.gradients(self.loss, tf.compat.v1.trainable_variables())
        W = tf.compat.v1.trainable_variables()[0]
        W_grad = grads[0]
        u = W @ self.v 
        self.v = tf.transpose(W) @ u 
        sigma = tf.norm(u, 2) / tf.norm(self.v, 2)
        reg_value = sigma * (u @ tf.transpose(self.v))
        W_grad += self.reg_constant * reg_value
        grads[0] = W_grad
         
        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.w_grad = grads[0]
        self.w = tf.compat.v1.trainable_variables()[0]

        self.train_op = self.optimizer.apply_gradients(zip(grads, tf.compat.v1.trainable_variables()))

class OrthogonalReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        super().__init__(config)

    def get_mlp(self):
        def orthogonal_reg(W):
            orthog_term = tf.math.reduce_sum(tf.abs(W @ tf.transpose(W) - tf.eye(W.shape.as_list()[0])))
            return self.reg_constant * orthog_term

        weights = np.load(str(init_weights_path))
        return tf.keras.layers.Dense(10, \
            activation='softmax',\
            kernel_regularizer=orthogonal_reg, \
            kernel_initializer=tf.constant_initializer(weights))

class L2Reg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        super().__init__(config)

    def get_mlp(self):
        weights = np.load(str(init_weights_path))
        def L2_reg(W):
            norm = tf.norm(W, 2)
            return self.reg_constant * norm

        return tf.keras.layers.Dense(10, \
            activation='softmax',\
            kernel_initializer=tf.constant_initializer(weights), \
            kernel_regularizer=L2_reg)

class L1Reg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        super().__init__(config)
    
    def get_mlp(self):
        weights = np.load(str(init_weights_path))
        def L1_reg(W):
            norm = tf.norm(W, 1)
            return self.reg_constant * norm

        return tf.keras.layers.Dense(10, \
            activation='softmax',\
            kernel_initializer=tf.constant_initializer(weights), \
            kernel_regularizer=L1_reg)


# didn't perform well 
class LipschitzReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        super().__init__(config)

    def build_graph(self):
        self.x_data = tf.placeholder(np.float32, [None, 784])
        self.y_data = tf.placeholder(np.float32, [None, 10])
        
        iterator = self.build_datapipeline()
        xb, yb = iterator.get_next()

        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        # lipschitz regularization
        grads = tf.gradients(self.loss, tf.compat.v1.trainable_variables())
        lipschitz_reg = tf.reduce_mean([(tf.norm(g, 2) - 1.) ** 2 for g in grads])
        self.loss += self.reg_constant * lipschitz_reg

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.w_grad = tf.gradients(self.loss, tf.compat.v1.trainable_variables())[0]
        self.w = tf.compat.v1.trainable_variables()[0]

        self.train_op = self.optimizer.minimize(self.loss)

# def get_model(is_training=True, dropout=0.):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(784,)))
#     if dropout > 0.:
#         model.add(tf.keras.layers.Dropout(dropout, training=is_training))
#     model.add(tf.keras.layers.Dense(10))
#     return model