#%%
import tensorflow.compat.v1 as tf 
import numpy as np 
from writers import NeptuneWriter

#%%
def get_train_test(batch_size=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

    x_train = x_train / 255.
    x_test = x_test / 255.

    # one-hot encoding 
    oh = np.zeros((y_train.size, 10))
    oh[np.arange(y_train.size), y_train[:, 0]] = 1 
    y_train = oh

    oh = np.zeros((y_test.size, 10))
    oh[np.arange(y_test.size), y_test[:, 0]] = 1 
    y_test = oh

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    y_train = y_train[:-10000]
    x_train = x_train[:-10000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

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
            tf.keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding='same'),
            
            tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation="relu", padding='same'), 
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2),

            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2), 

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128, activation="relu"), 
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
            .shuffle(10000)\
            .cache()\
            .batch(self.batch_size)
        self.dataset_iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                  dataset.output_shapes)
        self.dataset_iterator_init = self.dataset_iterator.make_initializer(dataset)
    
    def build_graph(self):
        self.build_datapipeline()

        # model evaluation 
        xb, yb = self.dataset_iterator.get_next()
        xb.set_shape([None, 32, 32, 3])

        self.logits = self.model(xb)
        self.loss = self.loss_func(yb, self.logits)

        # add layer losses (L1, L2, etc.)
        if self.layer_regularization: 
            for layer in self.layers: 
                self.loss += tf.math.reduce_sum(layer.losses)

        self.grads = tf.gradients(self.loss, tf.trainable_variables())

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(self.logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.minimize(self.loss)

def init_metrics():
    custom_metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    return custom_metrics

def mean_over_dict(custom_metrics):
    mean_metrics = {}
    for k in custom_metrics.keys(): 
        mean_metrics[k] = np.mean(custom_metrics[k])
    return mean_metrics

#%%
def train(trainer):
    # for early stopping 
    require_improvement = 20
    last_improvement = 0 
    stop = False 

    with tf.Session() as sess: 
        best_sess = sess
        best_score = 0. 

        sess.run([tf.global_variables_initializer(), \
            tf.local_variables_initializer()])

        for e in range(config['epochs']):
            metrics = init_metrics()
            
            # training 
            sess.run(trainer.acc_initializer) # reset accuracy metric
            sess.run(trainer.dataset_iterator_init, \
                feed_dict={trainer.x_data: x_train, trainer.y_data: y_train})
            try: 
                while True:
                    _, loss, _ = \
                        sess.run([trainer.train_op, trainer.loss,\
                            trainer.acc_op])  
                    metrics['train_loss'].append(loss)
                    if trial_run: break 
            except tf.errors.OutOfRangeError: pass 
            train_acc = sess.run(trainer.acc)
            metrics['train_acc'] = [train_acc]

            # validation 
            try: 
                sess.run(trainer.acc_initializer) # reset accuracy metric
                sess.run(trainer.dataset_iterator_init, feed_dict={
                    trainer.x_data: x_val, 
                    trainer.y_data: y_val, 
                    trainer.is_training: False})
                while True:
                    loss, _ = sess.run([trainer.loss, trainer.acc_op])        
                    metrics['val_loss'].append(loss)
                    if trial_run: break 
            except tf.errors.OutOfRangeError: pass 
            val_acc = sess.run(trainer.acc)
            metrics['val_acc'] = [val_acc]

            # early stopping
            ## https://stackoverflow.com/questions/46428604/how-to-implement-early-stopping-in-tensorflow
            if val_acc > best_score:
                best_sess = sess # save session
                best_score = val_acc
                last_improvement = 0
            else:
                last_improvement += 1
            if last_improvement > require_improvement:
                # Break out from the loop.
                stop = True

            epoch_metrics = mean_over_dict(metrics)
            writer.write(epoch_metrics, e)

            print('{}: {} {}'.format(e, train_acc, val_acc))
    
            if stop: 
                print('Early stopping...')
                break 

        # test set 
        try: 
            sess = best_sess # restore session with the best score
            sess.run(trainer.acc_initializer) # reset accuracy metric
            sess.run(trainer.dataset_iterator_init, feed_dict={
                trainer.x_data: x_test, 
                trainer.y_data: y_test, 
                trainer.is_training: False})
            while True:
                _ = sess.run([trainer.acc_op])
                if trial_run: break 
        except tf.errors.OutOfRangeError: pass 
        test_acc = sess.run(trainer.acc)
        writer.write({'test_acc': test_acc}, e+1)

    return trainer

#%%
(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_test()

# %%
trial_run = True 
config = {
    'batch_size': 32 if not trial_run else 2,
    'epochs': 200 if not trial_run else 1,
    'reg_constant': 0.01,
    'dropout_constant': 0.3,
}
trainers = [Baseline]
configs = [config]

writer = NeptuneWriter('gebob19/672-cifar')

for config, trainer_class in zip(configs, trainers): 
    config['experiment_name'] = trainer_class.__name__
    if not trial_run:
        writer.start(config)

    tf.reset_default_graph()
    full_trainer = train(trainer_class(config))
    
    writer.fin()

# %%

# %%
