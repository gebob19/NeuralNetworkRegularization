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

    # one-hot encoding 
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
batch_size = 32 
epochs = 5
reg_constant = 10.
dropout_constant = 0.3

(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_test()

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
