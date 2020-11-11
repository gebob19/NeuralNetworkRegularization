#%%
import tensorflow as tf 
import imageio
import numpy as np
import cv2 
from writers import NeptuneWriter
print(tf.__version__)

#%%
IMAGE_SIZE_H, IMAGE_SIZE_W = 224, 224
BATCH_SIZE = 2
PREFETCH_BUFFER = BATCH_SIZE
NUM_CLASSES = 2000

def mp4_2_numpy(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    data = np.stack(list(vid.iter_data()))
    return data 

def py_line2example(line):
    d = line.numpy().decode("utf-8").split(' ')
    fn, label = d[0], int(d[1])
    vid = mp4_2_numpy(fn)
    t, h, w, _ = vid.shape 
    
    # sample 10 random frames  -- fps == 24 
    sampled_frame_idxs = np.linspace(0, t-1, num=10, dtype=np.int32)
    vid = vid[sampled_frame_idxs]

    if h < IMAGE_SIZE_H or w < IMAGE_SIZE_W: 
        vid = np.stack([cv2.resize(frame, (IMAGE_SIZE_W, IMAGE_SIZE_H)) for frame in vid])
    else: 
        # crop to IMAGE_SIZE x IMAGE_SIZE
        h_start = (h - IMAGE_SIZE_H) // 2
        w_start = (w - IMAGE_SIZE_W) // 2
        vid = vid[:, h_start: h_start + IMAGE_SIZE_H, w_start: w_start + IMAGE_SIZE_W, :]
    
    # transpose to (c, h, w, temporal)
    vid = vid.transpose((-1, 1, 2, 0))
    vid = vid / 255. * 2 - 1

    # one-hot encode label 
    oh = np.zeros((NUM_CLASSES,))
    oh[label] = 1 

    return vid, oh
    
def line2example(x):
    image, label = tf.py_function(py_line2example, [x], [tf.float32, tf.int32])
    return image, label

def mean_over_dict(custom_metrics):
    mean_metrics = {}
    for k in custom_metrics.keys(): 
        mean_metrics[k] = np.mean(custom_metrics[k])
    return mean_metrics

#%%
class Dropout(Baseline):
    def __init__(self, config):
        ## include dropout before dense layer 
        self.dropout = tf.keras.layers.Dropout(config['DROPOUT_CONSTANT'])
        self.dense1 = tf.keras.layers.Dense(4096)
        self.dense2 = tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax')
        super().__init__(config)

    def get_layers(self, config):
        return [
            tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((1, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),
            
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Flatten(), 
            # cut out dropout layer from this so we can include training flag 
        ]
    
    def model(self, x):
        x = super().model(x)
        x = self.dropout(x, training=self.is_training)
        x = self.dense1(x)
        x = self.dense2(x)
        return x 

class Baseline():
    def __init__(self, config):
        self.layers = self.get_layers(config)

        self.optimizer = tf.train.AdamOptimizer(1e-2)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.is_training = tf.placeholder_with_default(True, shape=())
        self.build_graph()

    def get_layers(self, config):
        return [
            tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((1, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),
            
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(4096),
            tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax'),
        ]

    def model(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x 

    def build_datapipeline(self):
        train_dataset = tf.data.TextLineDataset(['train.txt'])\
            .map(line2example)\
            .prefetch(PREFETCH_BUFFER)\
            .batch(BATCH_SIZE)

        val_dataset = tf.data.TextLineDataset(['val.txt'])\
            .map(line2example)\
            .prefetch(PREFETCH_BUFFER)\
            .batch(BATCH_SIZE)
        
        test_dataset = tf.data.TextLineDataset(['test.txt'])\
            .map(line2example)\
            .prefetch(PREFETCH_BUFFER)\
            .batch(BATCH_SIZE)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.train_handle = self.train_iterator.string_handle()

        self.val_iterator = val_dataset.make_initializable_iterator()
        self.val_handle = self.val_iterator.string_handle()

        self.test_iterator = test_dataset.make_initializable_iterator()
        self.test_handle = self.test_iterator.string_handle()

        self.handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
        dataset_iterator = tf.data.Iterator.from_string_handle(self.handle_flag, 
            tf.compat.v1.data.get_output_types(train_dataset), 
            tf.compat.v1.data.get_output_shapes(train_dataset))

        return dataset_iterator
    
    def build_graph(self):
        dataset_iterator = self.build_datapipeline()

        # model evaluation 
        xb, yb = dataset_iterator.get_next()
        xb.set_shape([None, 3, IMAGE_SIZE_H, IMAGE_SIZE_W, 10])

        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        self.acc, self.acc_op = tf.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.minimize(self.loss)

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
}
config['experiment_name'] = type(trainer).__name__
# writer.start(config)
# trainer = Baseline(config)
trainer = Dropout(config)

with tf.Session() as sess:
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

#%%
var = tf.trainable_variables()[0]

#%%
#%%
# %%
