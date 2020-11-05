#%%
import numpy as np 
import tensorflow as tf 

from tqdm import tqdm 

from models import * 
from writers import NeptuneWriter

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.enable_eager_execution()
print(tf.__version__)

#%%
def get_train_test(batch_size=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.reshape(x_train, (-1, 784)).astype(np.float32)
    x_test = np.reshape(x_test, (-1, 784)).astype(np.float32)

    oh = np.zeros((y_train.size, 10))
    oh[np.arange(y_train.size), y_train] = 1 
    y_train = oh

    oh = np.zeros((y_test.size, 10))
    oh[np.arange(y_test.size), y_test] = 1 
    y_test = oh

    # Reserve 10,000 samples for validation.
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    y_train = y_train[:-10000]
    x_train = x_train[:-10000]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def mean_over_dict(custom_metrics):
    mean_metrics = {}
    for k in custom_metrics.keys(): 
        mean_metrics[k] = np.mean(custom_metrics[k])
    return mean_metrics

def init_metrics():
    custom_metrics = {
        'w_norm': [],
        'w_mean': [],
        'w_var': [],
        'w_rank': [],
        'wg_norm': [],
        'wg_mean': [],
        'wg_var': [],
        'loss': []
    }
    return custom_metrics

def record_metrics(w, w_grad, loss, metrics): 
    metrics['w_norm'] = np.linalg.norm(w)
    metrics['wg_norm'] = np.linalg.norm(w_grad)
    metrics['w_mean'] = np.mean(w)
    metrics['wg_mean'] = np.mean(w_grad)
    metrics['w_var'] = np.var(w)
    metrics['wg_var'] = np.var(w_grad)
    metrics['w_rank'] = np.linalg.matrix_rank(wt)
    metrics['loss'] = loss 

def get_model(dropout=0.):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(784,)))
    if dropout > 0.:
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(10))
    return model

#%%
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

def train(trainer):
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for e in range(epochs):
            metrics = init_metrics()
            # training 
            sess.run(trainer.dset_init, \
                feed_dict={trainer.x_data: x_train, trainer.y_data: y_train})
            try: 
                while True:
                    _, losst, _, wt, w_gradt = \
                        sess.run([trainer.train_op, trainer.loss, \
                            trainer.acc_op, trainer.w, trainer.w_grad])  
                    
                    record_metrics(wt, w_gradt, losst, metrics)
            except tf.errors.OutOfRangeError: pass 

            train_acc = sess.run(trainer.acc)
            metrics['train_acc'] = [train_acc]

            # validation 
            try: 
                sess.run(trainer.acc_initializer) # reset accuracy metric
                sess.run(trainer.dset_init, feed_dict={trainer.x_data: x_val, trainer.y_data: y_val})
                while True:
                    losst, _ = sess.run([trainer.loss, trainer.acc_op])        
            except tf.errors.OutOfRangeError: pass 

            val_acc = sess.run(trainer.acc)
            metrics['val_acc'] = [val_acc]

            epoch_metrics = mean_over_dict(metrics)
            writer.write(epoch_metrics, e)
        
            print('{}: {} {}'.format(e, train_acc, val_acc))
    writer.fin()


#%%
(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_test()
batch_size = 32 
epochs = 5
reg_constant = 10.
dropout_constant = 0.3

tf.reset_default_graph()
# trainer = Baseline()
trainer = LipschitzReg(reg_constant)
# trainer = DropoutReg(dropout_constant)
# trainer = SpectralReg(reg_constant)
# trainer = OrthogonalReg(reg_constant)

writer = NeptuneWriter('gebob19/672')
# writer.start({'experiment_name': type(trainer)})
train(trainer)

# %%
