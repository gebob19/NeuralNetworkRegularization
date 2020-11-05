import tensorflow as tf 

def get_model(dropout=0.):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(784,)))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(10))
    return model

class Baseline():
    def __init__(self):
        self.optimizer = tf.train.GradientDescentOptimizer(1e-2)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.build_graph()

    def build_datapipeline(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data))\
            .batch(batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                dataset.output_shapes)
        self.dset_init = iterator.make_initializer(dataset)
        return iterator

    def get_model(self):
        return get_model()

    def build_graph(self):
        self.x_data = tf.placeholder(np.float32, [None, 784])
        self.y_data = tf.placeholder(np.float32, [None, 10])

        model = self.get_model()

        iterator = self.build_datapipeline()
        xb, yb = iterator.get_next()

        logits = model(xb)
        self.loss = loss_func(yb, logits)

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.w_grad = tf.gradients(self.loss, model.trainable_weights)[0]
        self.w = model.trainable_weights[0]

        self.train_op = optimizer.minimize(self.loss)

class LipschitzReg(Baseline):
    def __init__(self, reg_constant):
        self.reg_constant = reg_constant
        super().__init__()

    def build_graph(self):
        self.x_data = tf.placeholder(np.float32, [None, 784])
        self.y_data = tf.placeholder(np.float32, [None, 10])
        
        model = self.get_model()

        iterator = self.build_datapipeline()
        xb, yb = iterator.get_next()

        logits = model(xb)
        self.loss = loss_func(yb, logits)

        # lipschitz regularization
        grads = tf.gradients(self.loss, model.trainable_weights)
        lipschitz_reg = tf.reduce_mean([(tf.norm(g, 2) - 1.) ** 2 for g in grads])
        self.loss += self.reg_constant * lipschitz_reg

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.w_grad = tf.gradients(self.loss, model.trainable_weights)[0]
        self.w = model.trainable_weights[0]

        self.train_op = optimizer.minimize(self.loss)

class DropoutReg(Baseline):
    def __init__(self, dropout_constant): 
        self.dropout_constant = dropout_constant
        super().__init__()

    def get_model(self):
        return get_model(self.dropout_constant)

class SpectralReg(Baseline):
    def __init__(self, reg_constant):
        self.reg_constant = reg_constant
        self.v = tf.random.normal((10, 1), mean=0., stddev=1.)
        super().__init__()

    def train(self, xb, yb):
        self.x_data = tf.placeholder(np.float32, [None, 784])
        self.y_data = tf.placeholder(np.float32, [None, 10])

        model = self.get_model()

        iterator = self.build_datapipeline()
        xb, yb = iterator.get_next()

        logits = model(xb)
        self.loss = loss_func(yb, logits)

        # apply spectral norm reg. 
        grads = tf.gradients(self.loss, model.trainable_weights)
        W = model.trainable_weights[0]
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
        self.w = model.trainable_weights[0]

        self.train_op = self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

class OrthogonalReg(Baseline):
    def __init__(self, reg_constant):
        self.reg_constant = reg_constant
        super().__init__()

    def get_model(self):
        def orthogonal_reg(W):
            orthog_term = tf.abs(W @ tf.transpose(W) - tf.eye(W.shape.as_list()[0])).sum()
            return self.reg_constant * orthog_term

        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(784,)))
        model.add(tf.keras.layers.Dense(10, kernel_regularizer=orthogonal_reg))
        return model 