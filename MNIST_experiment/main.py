#%%
import numpy as np 
import tensorflow as tf 

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from tqdm import tqdm 

from models import * 

import sys 
import pathlib 
# for LOCAL use 
sys.path.append(str(pathlib.Path.home()/'Documents/gradschool/672/project/regularization_project'))
# for COLAB use 
sys.path.append('/content/')
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

def log_weights(W, writer):
    x = TSNE(n_components=2).fit_transform(W.T)
    for i, xt in enumerate(x): 
        plt.scatter(xt[0], xt[1], label=str(i))
    plt.legend()
    if writer.has_started: 
        writer.experiment.log_image('TSNE_weights', plt.gcf())
    plt.clf()

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
        'largest_singular_value': [],
        'smallest_singular_value': [],
        'sum_singular_value': [],
        'loss': [],
    }
    return custom_metrics

def record_metrics(w, w_grad, loss, metrics): 
    metrics['w_norm'].append(np.linalg.norm(w))
    metrics['wg_norm'].append(np.linalg.norm(w_grad))
    metrics['w_mean'].append(np.mean(w))
    metrics['wg_mean'].append(np.mean(w_grad))
    metrics['w_var'].append(np.var(w))
    metrics['wg_var'].append(np.var(w_grad))
    metrics['w_rank'].append(np.linalg.matrix_rank(w))

    singular_values = np.linalg.svd(w, compute_uv=False)
    metrics['largest_singular_value'].append(singular_values.max())
    metrics['smallest_singular_value'].append(singular_values.min())
    metrics['sum_singular_value'].append(singular_values.sum())

    metrics['loss'].append(loss)

def train(trainer):
    # for early stopping 
    require_improvement = 10
    stop = False 

    with tf.compat.v1.Session() as sess: 
        best_sess = sess
        best_score = 0. 
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        for e in range(config['epochs']):
            metrics = init_metrics()
            
            # training 
            sess.run(trainer.acc_initializer) # reset accuracy metric
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
                sess.run(trainer.dset_init, feed_dict={
                    trainer.x_data: x_val, 
                    trainer.y_data: y_val, 
                    trainer.is_training: False})
                while True:
                    losst, _ = sess.run([trainer.loss, trainer.acc_op])        
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
            sess.run(trainer.dset_init, feed_dict={
                trainer.x_data: x_test, 
                trainer.y_data: y_test, 
                trainer.is_training: False})
            while True:
                _ = sess.run([trainer.acc_op])
        except tf.errors.OutOfRangeError: pass 
        test_acc = sess.run(trainer.acc)
        writer.write({'test_acc': test_acc}, e+1)

        W = sess.run(trainer.w)
    return trainer, W 

#%%
config = {
    'batch_size': 32,
    'epochs': 1,
    'reg_constant': 0.01,
    'dropout_constant': 0.3,
}
(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_test()

# default configs 
trainers = [Baseline, L1Reg, L2Reg, DropoutReg, SpectralReg, OrthogonalReg]
configs = [config.copy(), config.copy(), config.copy(), config.copy(), config.copy(), config.copy()]

# variations of regularization/dropout parameters 
new_trainers, new_configs = [], []
for config, trainer_class in zip(configs, trainers):
    if trainer_class.__name__ == 'Baseline': 
        continue

    if trainer_class.__name__ != 'DropoutReg':
        new_confg = config.copy()
        new_confg['reg_constant'] *= 10
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

        new_confg = config.copy()
        new_confg['reg_constant'] *= 100
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

        new_confg = config.copy()
        new_confg['reg_constant'] /= 10
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

        new_confg = config.copy()
        new_confg['reg_constant'] /= 100
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)
    else: 
        new_confg = config.copy()
        new_confg['dropout_constant'] = 0.5
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

        new_confg = config.copy()
        new_confg['dropout_constant'] = 0.8
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

        new_confg = config.copy()
        new_confg['dropout_constant'] = 0.1
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)
        
        new_confg = config.copy()
        new_confg['dropout_constant'] = 0.2
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

trainers += new_trainers
configs += new_configs

print(trainers[-1], configs[-1])

writer = NeptuneWriter('gebob19/672-mnist')

trainers = [Baseline]
configs = [config]

for config, trainer_class in zip(configs, trainers): 
    config['experiment_name'] = trainer_class.__name__
    # writer.start(config)

    tf.reset_default_graph()
    full_trainer, W = train(trainer_class(config))
    log_weights(W, writer)
    
    writer.fin()

print('Complete!')
