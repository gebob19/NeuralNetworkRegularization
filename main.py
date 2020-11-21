#%%
import tensorflow.compat.v1 as tf 
import numpy as np 
from writers import NeptuneWriter
from models import *

#%%
def get_train_test():
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
    require_improvement = 15
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
            sess.run(trainer.iterator_init, \
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
                sess.run(trainer.iterator_init, feed_dict={
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

            print('{}: {:.2f} acc: {:.2f} {:.2f}'.format(e, epoch_metrics['train_loss'], train_acc, val_acc))
    
            if stop: 
                print('Early stopping...')
                break 

        # test set 
        try: 
            sess = best_sess # restore session with the best score
            sess.run(trainer.acc_initializer) # reset accuracy metric
            sess.run(trainer.iterator_init, feed_dict={
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
trial_run = True
config = {
    'batch_size': 64 if not trial_run else 2,
    'epochs': 200 if not trial_run else 1,
    'reg_constant': 0.01,
    'dropout_constant': 0.3,
    'kernel_regularization': True, 
    'dense_regularization': True, 
}

(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_train_test()

# default configs 
trainers = [Baseline, Dropout, SpectralReg, OrthogonalReg, L2Reg, L1Reg]
configs = [config.copy(), config.copy(), config.copy(), config.copy(), config.copy(), config.copy()]

# include kernel + dense regularization 
d_config = config.copy()
d_config['kernel_regularization'] = False 
k_config = config.copy()
k_config['dense_regularization'] = False 

new_trainers = []
new_configs = []

# variations of regularization/dropout parameters 
new_trainers, new_configs = [], []
for config, trainer_class in zip(configs, trainers):
    if trainer_class.__name__ == 'Baseline': 
        continue

    if trainer_class.__name__ == 'OrthogonalReg':
        new_confg = config.copy()
        new_confg['reg_constant'] = 0.1
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)
        
        new_confg = config.copy()
        new_confg['reg_constant'] = 0.001
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)
        
        new_confg = config.copy()
        new_confg['reg_constant'] = 0.0001
        new_configs.append(new_confg)
        new_trainers.append(trainer_class)

    elif trainer_class.__name__ == 'Dropout':
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

    else: 
        new_confg = config.copy()
        new_confg['reg_constant'] *= 10
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
        
trainers += new_trainers
configs += new_configs

if trial_run:
    trainers = [SpectralReg]
    configs = [config]

writer = NeptuneWriter('gebob19/672-cifar')

for config, trainer_class in zip(configs, trainers): 
    config['experiment_name'] = trainer_class.__name__
    if not trial_run:
        writer.start(config)

    tf.reset_default_graph()
    full_trainer = train(trainer_class(config))
    
    writer.fin()

print('completed!')