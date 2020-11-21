import tensorflow.compat.v1 as tf 
import numpy as np 

# %%
class Baseline():
    def __init__(self, config):
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.batch_size = config['batch_size']
        self.layers = self.get_layers(config)
        self.layer_regularization = False 

        self.build_graph()

    def get_layers(self, config): 
        return [
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(64, activation="relu"), 
            tf.keras.layers.Dense(10, activation='softmax'), 
        ]
        # return [
        #     tf.keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding='same'),
            
        #     tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation="relu", padding='same'), 
        #     tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
        #     tf.keras.layers.MaxPool2D(2, padding='same'),

        #     tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        #     tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
        #     tf.keras.layers.MaxPool2D(2, padding='same'), 

        #     tf.keras.layers.Flatten(), 
        #     tf.keras.layers.Dense(128, activation="relu"), 
        #     tf.keras.layers.Dense(10, activation='softmax'), 
        # ]
    
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

        # # add layer losses (L1, L2, etc.)
        # if self.layer_regularization: 
        #     for layer in self.layers: 
        #         self.loss += tf.math.reduce_sum(layer.losses)

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(self.logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.minimize(self.loss)
