#%%
import tensorflow as tf 
import numpy as np
import imageio
import cv2 
from writers import NeptuneWriter
from models import * 
from config import * 

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#%%
def mean_over_dict(custom_metrics):
    mean_metrics = {}
    for k in custom_metrics.keys(): 
        mean_metrics[k] = np.mean(custom_metrics[k])
    return mean_metrics

def train(trainer):
    with tf.compat.v1.Session() as sess:
        best_sess = sess
        best_score = 0. 
        last_improvement = 0
        stop = False 

        import time 
        start = time.time()
        sess.run([tf.compat.v1.global_variables_initializer(), \
            tf.compat.v1.local_variables_initializer()])
        print(time.time() - start)
        1/0

        train_handle_value, val_handle_value, test_handle_value = \
            sess.run([trainer.train_handle, trainer.val_handle, trainer.test_handle])

        for e in range(EPOCHS):
            metrics = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

            sess.run([trainer.train_iterator.initializer, \
                trainer.val_iterator.initializer, trainer.test_iterator.initializer])

            # training 
            try: 
                sess.run(trainer.acc_initializer) # reset accuracy metric
                while True:
                    _, loss, _ = sess.run([trainer.train_op, trainer.loss, \
                                trainer.acc_op], \
                                feed_dict={trainer.handle_flag: train_handle_value,
                                trainer.is_training: True})
                    metrics['train_loss'].append(loss)
                    
                    if TRIAL_RUN: break 
            except tf.errors.OutOfRangeError: pass 
            metrics['train_acc'] = [sess.run(trainer.acc)]

            # validation 
            try: 
                sess.run(trainer.acc_initializer) # reset accuracy metric
                while True:
                    loss, _ = sess.run([trainer.loss, trainer.acc_op], \
                        feed_dict={trainer.handle_flag: val_handle_value, 
                        trainer.is_training: False})        
                    metrics['val_loss'].append(loss)
                    
                    if TRIAL_RUN: break 
            except tf.errors.OutOfRangeError: pass 
            val_acc = sess.run(trainer.acc)
            metrics['val_acc'] = [val_acc]

            # early stopping
            ## https://stackoverflow.com/questions/46428604/how-to-implement-early-stopping-in-tensorflow
            if val_acc > best_score:
                best_sess = sess # save session
                best_score = val_acc
            else:
                last_improvement += 1
            if last_improvement > REQUIRED_IMPROVEMENT:
                # Break out from the loop.
                stop = True

            mean_metrics = mean_over_dict(metrics)
            writer.write(mean_metrics, e)

            print("{} {}".format(e, mean_metrics))

            if stop: 
                print('Early stopping...')
                break 

        # test 
        try: 
            sess = best_sess # restore session with the best score
            sess.run(trainer.acc_initializer) # reset accuracy metric
            while True:
                sess.run([trainer.acc_op], \
                    feed_dict={
                        trainer.handle_flag: test_handle_value, 
                        trainer.is_training: False})        
                if TRIAL_RUN: break 
        except tf.errors.OutOfRangeError: pass 

        test_acc = sess.run(trainer.acc)
        writer.write({'test_acc': test_acc}, e+1)
        print('test_accuracy: ', test_acc)

        writer.fin()
    return trainer 

#%%
EPOCHS = 100
REQUIRED_IMPROVEMENT = 10
TRIAL_RUN = True

writer = NeptuneWriter('gebob19/672-asl')
config = {
    'EPOCHS': EPOCHS if not TRIAL_RUN else 1,
    'BATCH_SIZE': BATCH_SIZE if not TRIAL_RUN else 2, 
    'IMAGE_SIZE_H': IMAGE_SIZE_H ,
    'IMAGE_SIZE_W': IMAGE_SIZE_W,
    'PREFETCH_BUFFER': PREFETCH_BUFFER if not TRIAL_RUN else 2,
    'NUM_CLASSES': NUM_CLASSES,
    'DROPOUT_CONSTANT': 0.5,
    'REG_CONSTANT': 0.01, 
    'REQUIRED_IMPROVEMENT': REQUIRED_IMPROVEMENT,
 }

# default configs 
trainers = [Baseline, L1Reg, L2Reg, Dropout, SpectralReg, OrthogonalReg]
configs = [config.copy(), config.copy(), config.copy(), config.copy(), config.copy(), config.copy()]

if TRIAL_RUN:
    trainers = [OrthogonalReg]
    configs = [config]

for config, trainer_class in zip(configs, trainers): 
    config['experiment_name'] = trainer_class.__name__
    if not TRIAL_RUN:
        writer.start(config)

    tf.compat.v1.reset_default_graph()
    full_trainer = train(trainer_class(config))
    log_weights(W, writer)
    
    writer.fin()

print('Complete!')

# %%

