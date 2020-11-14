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

#%%
tf.reset_default_graph() # reset!
EPOCHS = 1
REQUIRED_IMPROVEMENT = 10
trial_run = True

writer = NeptuneWriter('gebob19/672-asl')
config = {
    'EPOCHS': EPOCHS,
    'BATCH_SIZE': BATCH_SIZE, 
    'IMAGE_SIZE_H': IMAGE_SIZE_H ,
    'IMAGE_SIZE_W': IMAGE_SIZE_W,
    'BATCH_SIZE': BATCH_SIZE,
    'PREFETCH_BUFFER': PREFETCH_BUFFER,
    'NUM_CLASSES': NUM_CLASSES,
    'DROPOUT_CONSTANT': 0.5,
    'REG_CONSTANT': 0.01, 
    'REQUIRED_IMPROVEMENT': 10,
}
# writer.start(config)

# trainer = Baseline(config)
# trainer = Dropout(config)
# trainer = SpectralReg(config)
trainer = OrthogonalReg(config)
# trainer = L2Reg(config)
# trainer = L1Reg(config)

config['experiment_name'] = type(trainer).__name__

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    best_sess = sess
    best_score = 0. 
    last_improvement = 0
    stop = False 

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
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
                
                if trial_run: break 
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
                
                if trial_run: break 
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
            if trial_run: break 
    except tf.errors.OutOfRangeError: pass 

    test_acc = sess.run(trainer.acc)
    writer.write({'test_acc': test_acc}, e+1)
    print('test_accuracy: ', test_acc)

    writer.fin()

# #%%
# x = np.random.randn(3, 10, 223, 223, 4)
# conv = tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same')
# conv(x).shape

# #%%
# W = conv.weights[0]

# #%%
# I = np.zeros((3, 3, 3))
# I[1, 1, 1] = 1
# I3d = np.stack([[np.stack([I] * W.shape[-2].value)] * W.shape[-1].value])[0].transpose((2, 3, 4, 1, 0))

# %%
