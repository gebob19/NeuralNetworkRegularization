import tensorflow.compat.v1 as tf 
import numpy as np 

# %%
class Baseline():
    def __init__(self, config):
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.batch_size = config['batch_size']
        self.layers = self.get_layers(config)
        self.layer_regularization = self.get_layer_regularization_flag() 

        self.build_graph()

    def get_layer_regularization_flag(self):
        return False 

    def get_layers(self, config): 
        return [
            tf.keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding='same'),
            
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'), 
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same'),

            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same'), 

            tf.keras.layers.Conv2D(512, 3, activation="relu", padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same'), 

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128, activation="relu"), 
            tf.keras.layers.Dense(256, activation="relu"), 
            tf.keras.layers.Dense(10, activation='softmax'), 
        ]
    
    def model(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x 

    def build_datapipeline(self):
        self.x_data = tf.placeholder(np.float32, [None, 32, 32, 3])
        self.y_data = tf.placeholder(np.float32, [None, 10])
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data))\
            .batch(self.batch_size)

        self.dataset_iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                  dataset.output_shapes)
        self.iterator_init = self.dataset_iterator.make_initializer(dataset)
    
    def build_graph(self):
        self.build_datapipeline()

        # model evaluation 
        xb, yb = self.dataset_iterator.get_next()

        self.logits = self.model(xb)
        self.loss = self.loss_func(yb, self.logits)

        # add layer losses (L1, L2, etc.)
        if self.layer_regularization: 
            for layer in self.layers: 
                self.loss += tf.math.reduce_sum(layer.losses)

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(self.logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.minimize(self.loss)

class Dropout(Baseline):
    def __init__(self, config):
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        super().__init__(config)

    def get_layers(self, config): 
        return [
            ([tf.keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding='same'),],
            tf.keras.layers.Dropout(config['dropout_constant'])),
            
            ([tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'), 
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same')],
            tf.keras.layers.Dropout(config['dropout_constant'])),

            ([tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same')], 
            tf.keras.layers.Dropout(config['dropout_constant'])),
            
            ([tf.keras.layers.Conv2D(512, 3, activation="relu", padding='same'),
            tf.keras.layers.Conv2D(512, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same')], 
            tf.keras.layers.Dropout(config['dropout_constant'])),
        ]
    
    def model(self, x):
        for block, dropout in self.layers: 
            for layer in block: 
                x = layer(x)
            x = dropout(x, training=self.is_training)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x 

class SpectralReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        self.config = config
        super().__init__(config)

    def build_graph(self):
        self.build_datapipeline()

        xb, yb = self.dataset_iterator.get_next()
        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        self.variables = [(v, i) for i, v in enumerate(tf.trainable_variables()) if 'kernel' in v.name]
        # dont apply to last dense layer 
        self.variables.pop(-1)
        if not self.config['kernel_regularization']:
            self.variables = [(v, i) for v, i in self.variables if not 'conv2d' in v.name]
        if not self.config['dense_regularization']:
            self.variables = [(v, i) for v, i in self.variables if not 'dense' in v.name]
        self.vs = [tf.random.normal((v.shape.as_list()[-1], 1), mean=0., stddev=1.) for v, _ in self.variables]

        assert len(self.variables) > 0
        # spectral norm reg
        grads = tf.gradients(self.loss, tf.trainable_variables())
        new_vs = []
        for (var, idx), v in zip(self.variables, self.vs):
            original_shape = grads[idx].shape
            W_grad = tf.reshape(grads[idx], [-1, var.shape[-1]])
            W = tf.reshape(var, [-1, var.shape[-1]])

            u = W @ v
            v = tf.transpose(W) @ u 
            sigma = tf.norm(u, 2) / tf.norm(v, 2)
            reg_value = sigma * (u @ tf.transpose(v))
            W_grad += self.reg_constant * reg_value
            
            grads[idx] = tf.reshape(W_grad, original_shape)
            new_vs.append(v)
        self.vs = new_vs

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.apply_gradients(zip(grads, tf.trainable_variables()))

class OrthogonalReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['reg_constant']
        self.set_reg_method(config)
        super().__init__(config)

    def get_layer_regularization_flag(self):
        return True

    def set_reg_method(self, config):
        def orthogonal_reg(W):
            orthog_term = tf.math.reduce_sum(tf.abs(W @ tf.transpose(W) - tf.eye(W.shape.as_list()[0])))
            return self.reg_constant * orthog_term

        def orthogonal_flat_kernel_reg(W):
            W = tf.reshape(W, [-1, W.shape[-1]]) # flatten using same means as spectral 
            return orthogonal_reg(W)

        def orthogonal_kernel_reg(W):
            k = W.shape.as_list()[0]
            I = np.zeros((k, k))
            I[k // 2, k // 2] = 1
            I = tf.constant(I, dtype=tf.float32)
            tf_2deye = tf.transpose(tf.stack([tf.stack([I] * W.shape.as_list()[-2])] * W.shape.as_list()[-1]), perm=[2, 3, 1, 0])

            orthog_term = tf.math.reduce_sum(
                tf.abs(
                    W * tf.transpose(W, perm=[1, 0, 2, 3]) - tf_2deye
                )
            )
            return self.reg_constant * orthog_term

        self.dense_reg_method = orthogonal_reg if config['dense_regularization'] else None
        self.kernel_reg_method = orthogonal_flat_kernel_reg if config['kernel_regularization'] else None

    def get_layers(self, config):
        return [
            tf.keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method),
            
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method),
            tf.keras.layers.MaxPool2D(2, padding='same'),

            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method),
            tf.keras.layers.MaxPool2D(2, padding='same'), 

            tf.keras.layers.Conv2D(512, 3, activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method),
            tf.keras.layers.Conv2D(512, 3, activation="relu", padding='same', kernel_regularizer=self.kernel_reg_method),
            tf.keras.layers.MaxPool2D(2, padding='same'), 

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=self.dense_reg_method), 
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=self.dense_reg_method), 
            tf.keras.layers.Dense(10, activation='softmax'), 
        ]

class L2Reg(OrthogonalReg):
    def __init__(self, config):
        super().__init__(config)

    def set_reg_method(self, config):
        def L2_reg(W):
            norm = tf.norm(W, 2)
            return self.reg_constant * norm
        self.dense_reg_method = L2_reg if config['dense_regularization'] else None
        self.kernel_reg_method = L2_reg if config['kernel_regularization'] else None

class L1Reg(OrthogonalReg):
    def __init__(self, config):
        super().__init__(config)

    def set_reg_method(self, config):
        def L1_reg(W):
            norm = tf.norm(W, 1)
            return self.reg_constant * norm
        self.dense_reg_method = L1_reg if config['dense_regularization'] else None
        self.kernel_reg_method = L1_reg if config['kernel_regularization'] else None