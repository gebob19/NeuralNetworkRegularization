import tensorflow as tf 
import numpy as np 
import imageio
import cv2 
from config import * 

####### data loading helpers 

def mp4_2_numpy(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    data = np.stack(list(vid.iter_data()))
    return data 

def py_line2example(line, n_frames):
    d = line.numpy().decode("utf-8").split(' ')
    fn, label = '{}{}'.format(PATH2VIDEOS, d[0]), int(d[1])
    vid = mp4_2_numpy(fn)
    t, h, w, _ = vid.shape 
    
    # sample 10 random frames  -- fps == 24 
    sampled_frame_idxs = np.linspace(0, t-1, num=n_frames, dtype=np.int32)
    vid = vid[sampled_frame_idxs]

    if h < IMAGE_SIZE_H or w < IMAGE_SIZE_W: 
        vid = np.stack([cv2.resize(frame, (IMAGE_SIZE_W, IMAGE_SIZE_H)) for frame in vid])
    else: 
        # crop to IMAGE_SIZE x IMAGE_SIZE
        h_start = (h - IMAGE_SIZE_H) // 2
        w_start = (w - IMAGE_SIZE_W) // 2
        vid = vid[:, h_start: h_start + IMAGE_SIZE_H, w_start: w_start + IMAGE_SIZE_W, :]
    
    # squeeze
    if len(vid) == 1: 
        vid = vid[0]
    # T, H, W, C
    # vid = vid.transpose((-1, 1, 2, 0))
    vid = vid / 255. * 2 - 1

    # one-hot encode label 
    oh = np.zeros((NUM_CLASSES,))
    oh[label] = 1 

    return vid, oh
    
def line2example(x, n_frames=10):
    image, label = tf.py_function(py_line2example, [x, n_frames], [tf.float32, tf.int32])
    return image, label

######### TF Record helpers 

sequence_features = {
    'data': tf.FixedLenSequenceFeature([], dtype=tf.string)
}

context_features = {
    'filename': tf.io.FixedLenFeature([], tf.string),
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'temporal': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

@tf.function
def resize(x):
    return tf.image.resize(x, [IMAGE_SIZE_H, IMAGE_SIZE_W])

def _parse_image_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.
    context, sequence = tf.parse_single_sequence_example(
            example_proto, context_features=context_features, sequence_features=sequence_features)
    
    shape = (context['temporal'], context['height'], context['width'], context['depth'])

    # literally had to brute force this shit to get it working 
    video_data = tf.expand_dims(
        tf.image.decode_image(tf.gather(sequence['data'], [0])[0]), 0)
    i = tf.constant(1, dtype=tf.int32)
    cond = lambda i, _: tf.less(i, tf.cast(context['temporal'], tf.int32))
    def body(i, video_data):
        video3D = tf.gather(sequence['data'], [i])
        img_data = tf.image.decode_image(video3D[0]) 
        video_data = tf.concat([video_data, [img_data]], 0)
        return (tf.add(i, 1), video_data)

    _, video_data = tf.while_loop(cond, body, [i, video_data], 
        shape_invariants=[i.get_shape(), tf.TensorShape([None])])
    # use this to set the shape -- doesn't change anything 
    video3D = video_data
    video3D = tf.reshape(video3D, shape)
    video3D = tf.cast(video3D, tf.float32)

    # resize images in video
    video3D = tf.map_fn(resize, video3D, back_prop=False, parallel_iterations=10)
    # normalize to [0, 1]
    video3D = video3D / 255.

    label = context['label']
    label = tf.one_hot(label, NUM_CLASSES)

    return video3D, label, context['filename'], shape

def single_frame(video3D, label, fn, shape):
    # sample a single frame across temporal frames 
    idxs = tf.random.uniform((1,), minval=0,
            maxval=tf.cast(shape[0], tf.int32), 
            dtype=tf.int32)
    video3D = tf.squeeze(tf.gather(video3D, idxs))

    return video3D, label, fn, shape

####### trainer classes 

class Baseline():
    def __init__(self, config):
        self.layers = self.get_layers(config)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.is_training = tf.compat.v1.placeholder_with_default(True, shape=())
        self.layer_regularization = self.get_layer_regularization_flag()

        self.build_graph()

    def get_layer_regularization_flag(self):
        return False

    def get_layers(self, config):
        return [
            tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.MaxPool3D((1, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),
            
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax'),
        ]

    def model(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x 

    def set_input_shape(self, x):
        x.set_shape([None, 10, IMAGE_SIZE_H, IMAGE_SIZE_W, 3])

    def build_datapipeline(self):
        train_dataset = tf.data.TFRecordDataset(DATAFILE_PATH+'train.tfrecord', num_parallel_reads=3)\
            .map(_parse_image_function)\
            .prefetch(PREFETCH_BUFFER)\
            .batch(BATCH_SIZE)

        val_dataset = tf.data.TFRecordDataset(DATAFILE_PATH+'val.tfrecord', num_parallel_reads=3)\
            .map(_parse_image_function)\
            .prefetch(PREFETCH_BUFFER)\
            .batch(BATCH_SIZE)
        
        test_dataset = tf.data.TFRecordDataset(DATAFILE_PATH+'val.tfrecord', num_parallel_reads=3)\
            .map(_parse_image_function)\
            .prefetch(PREFETCH_BUFFER)\
            .batch(BATCH_SIZE)

        self.train_iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
        self.train_handle = self.train_iterator.string_handle()

        self.val_iterator = tf.compat.v1.data.make_initializable_iterator(val_dataset)
        self.val_handle = self.val_iterator.string_handle()

        self.test_iterator = tf.compat.v1.data.make_initializable_iterator(test_dataset)
        self.test_handle = self.test_iterator.string_handle()

        self.handle_flag = tf.compat.v1.placeholder(tf.string, [], name='iterator_handle_flag')
        self.dataset_iterator = tf.compat.v1.data.Iterator.from_string_handle(self.handle_flag, 
            tf.compat.v1.data.get_output_types(train_dataset), 
            tf.compat.v1.data.get_output_shapes(train_dataset))

        return self.dataset_iterator
    
    def build_graph(self):
        self.build_datapipeline()

        # model evaluation 
        xb, yb, _, _ = self.dataset_iterator.get_next()
        self.set_input_shape(xb)

        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        # add layer losses (L1, L2, etc.)
        if self.layer_regularization: 
            for layer in self.layers: 
                self.loss += tf.math.reduce_sum(layer.losses)

        self.grads = tf.gradients(self.loss, tf.compat.v1.trainable_variables())

        self.acc, self.acc_op = tf.compat.v1.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.compat.v1.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.minimize(self.loss)

class Baseline2D(Baseline):
    def __init__(self, config):
        # # simple resnet model 
        # self.model2d = tf.keras.applications.ResNet50(include_top=False, weights=None)
        # self.flatten = tf.keras.layers.Flatten()
        # self.dense1 = tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax')
        super().__init__(config)

    def get_layers(self, config):
        return [
            tf.keras.layers.Conv2D(64, 7, strides=(2, 2), activation="relu", padding='same'),
            
            tf.keras.layers.Conv2D(128, 3, strides=(2, 2), activation="relu", padding='same'), 
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same'),

            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding='same'),
            tf.keras.layers.MaxPool2D(2, padding='same'), 

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(128, activation="relu"), 
            tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax'), 
        ]

    # def model(self, x):
    #     x = self.model2d(x)
    #     x = self.flatten(x)
    #     x = self.dense1(x)
    #     return x 
    
    def build_datapipeline(self):
        self.x_data = tf.placeholder(np.float32, [None, 28, 28, 1])
        self.y_data = tf.placeholder(np.float32, [None, 10])
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data))\
            .batch(BATCH_SIZE)
        self.dataset_iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                  dataset.output_shapes)
        self.dset_init = self.dataset_iterator.make_initializer(dataset)

        return self.dataset_iterator

        # train_dataset = tf.data.TFRecordDataset(DATAFILE_PATH+'train.tfrecord')\
        #     .map(_parse_image_function)\
        #     .prefetch(PREFETCH_BUFFER)\
        #     .batch(BATCH_SIZE)

        # val_dataset = tf.data.TFRecordDataset(DATAFILE_PATH+'val.tfrecord')\
        #     .map(_parse_image_function)\
        #     .prefetch(PREFETCH_BUFFER)\
        #     .batch(BATCH_SIZE)
        
        # test_dataset = tf.data.TFRecordDataset(DATAFILE_PATH+'val.tfrecord')\
        #     .map(_parse_image_function)\
        #     .prefetch(PREFETCH_BUFFER)\
        #     .batch(BATCH_SIZE)

        # self.train_iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
        # self.train_handle = self.train_iterator.string_handle()

        # self.val_iterator = tf.compat.v1.data.make_initializable_iterator(val_dataset)
        # self.val_handle = self.val_iterator.string_handle()

        # self.test_iterator = tf.compat.v1.data.make_initializable_iterator(test_dataset)
        # self.test_handle = self.test_iterator.string_handle()

        # self.handle_flag = tf.compat.v1.placeholder(tf.string, [], name='iterator_handle_flag')
        # dataset_iterator = tf.compat.v1.data.Iterator.from_string_handle(self.handle_flag, 
        #     tf.compat.v1.data.get_output_types(train_dataset), 
        #     tf.compat.v1.data.get_output_shapes(train_dataset))
    
    def build_graph(self):
        self.build_datapipeline()

        # model evaluation 
        xb, yb = self.dataset_iterator.get_next()
        # xb.set_shape([None, IMAGE_SIZE_H, IMAGE_SIZE_W, 3])

        self.logits = self.model(self.xb)
        self.loss = self.loss_func(self.yb, self.logits)

        # add layer losses (L1, L2, etc.)
        if self.layer_regularization: 
            for layer in self.layers: 
                self.loss += tf.math.reduce_sum(layer.losses)

        self.grads = tf.gradients(self.loss, tf.compat.v1.trainable_variables())

        self.acc, self.acc_op = tf.compat.v1.metrics.accuracy(tf.argmax(self.yb, 1), tf.argmax(self.logits, 1), name='acc')
        self.acc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.compat.v1.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.minimize(self.loss)


class Dropout(Baseline):
    def __init__(self, config):
        ## include dropout before dense layer 
        self.dropout = tf.keras.layers.Dropout(config['DROPOUT_CONSTANT'])
        self.dense1 = tf.keras.layers.Dense(4096)
        self.dense2 = tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax')
        self.flatten = tf.keras.layers.Flatten()
        super().__init__(config)

    def get_layers(self, config):
        return [
            [tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((1, 2, 2), padding='same')],

            [tf.keras.layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')],

            [tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')],

            [tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')],
            
            [tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same'), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same')],
        ]
    
    def model(self, x):
        for block in self.layers: 
            # apply dropout after each block 
            for layer in block: 
                x = layer(x)
            x = self.dropout(x, training=self.is_training)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x 

class SpectralReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['REG_CONSTANT']
        self.variables = [(v, i) for i, v in enumerate(tf.trainable_variables()) if 'kernel' in v.name] 
        self.vs = [tf.random.normal((v.shape[-1], 1), mean=0., stddev=1.) for v, _ in self.variables]
        super().__init__(config)

    def build_graph(self):
        dataset_iterator = self.build_datapipeline()

        # model evaluation 
        xb, yb = dataset_iterator.get_next()
        xb.set_shape([None, 10, IMAGE_SIZE_H, IMAGE_SIZE_W, 3])

        logits = self.model(xb)
        self.loss = self.loss_func(yb, logits)

        # spectral norm reg
        grads = tf.gradients(self.loss, tf.compat.v1.trainable_variables())
        new_vs = []
        for (var, idx), v in zip(self.variables, self.vs):
            W_grad = tf.reshape(grads[idx], [-1, var.shape[-1]])
            W = tf.reshape(var, [-1, var.shape[-1]])

            u = W @ v
            v = tf.transpose(W) @ u 
            sigma = tf.norm(u, 2) / tf.norm(v, 2)
            reg_value = sigma * (u @ tf.transpose(v))
            W_grad += self.reg_constant * reg_value
            
            grads[idx] = W_grad
            new_vs.append(v)
        self.vs = new_vs

        self.acc, self.acc_op = tf.compat.v1.metrics.accuracy(tf.argmax(yb, 1), tf.argmax(logits, 1), name='acc')
        self.acc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="acc")
        self.acc_initializer = tf.compat.v1.variables_initializer(var_list=self.acc_vars)

        self.train_op = self.optimizer.apply_gradients(zip(grads, tf.compat.v1.trainable_variables()))

class OrthogonalReg(Baseline):
    def __init__(self, config):
        self.reg_constant = config['REG_CONSTANT']
        self.set_reg_method()
        super().__init__(config)

    def get_layer_regularization_flag(self):
        return True

    def set_reg_method(self):
        def orthogonal_reg(W):
            # W = tf.reshape(W, [-1, W.shape[-1]]) # flatten using same means as spectral 
            orthog_term = tf.math.reduce_sum(tf.abs(W @ tf.transpose(W) - tf.eye(W.shape.as_list()[0])))
            return self.reg_constant * orthog_term

        def orthogonal_kernel_reg(W):
            I = np.zeros((3, 3, 3))
            I[1, 1, 1] = 1
            I = tf.constant(I, dtype=tf.float32)
            tf_3deye = tf.transpose(tf.stack([tf.stack([I] * W.shape.as_list()[-2])] * W.shape.as_list()[-1]), perm=[2, 3, 4, 1, 0])

            orthog_term = tf.math.reduce_sum(
                tf.abs(
                    W * tf.transpose(W, perm=[0, 2, 1, -2, -1]) - tf_3deye
                )
            )
            return self.reg_constant * orthog_term

        self.dense_reg_method = orthogonal_reg
        self.kernel_reg_method = None

    def get_layers(self, config):
        return [
            tf.keras.layers.Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.MaxPool3D((1, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),
            
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same', kernel_regularizer=self.kernel_reg_method), 
            tf.keras.layers.MaxPool3D((2, 2, 2), padding='same'),

            tf.keras.layers.Flatten(), 
            tf.keras.layers.Dense(4096, kernel_regularizer=self.dense_reg_method),
            tf.keras.layers.Dense(config['NUM_CLASSES'], activation='softmax', kernel_regularizer=self.dense_reg_method),
        ]

class L2Reg(OrthogonalReg):
    def __init__(self, config):
        super().__init__(config)

    def set_reg_method(self):
        def L2_reg(W):
            norm = tf.norm(W, 2)
            return self.reg_constant * norm
        self.dense_reg_method = L2_reg
        self.kernel_reg_method = L2_reg

class L1Reg(OrthogonalReg):
    def __init__(self, config):
        super().__init__(config)

    def set_reg_method(self):
        def L1_reg(W):
            norm = tf.norm(W, 1)
            return self.reg_constant * norm
        self.dense_reg_method = L1_reg
        self.kernel_reg_method = L1_reg
