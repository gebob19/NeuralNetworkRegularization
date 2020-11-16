#%%
import tensorflow as tf 
import pathlib 
import json
import imageio 
import numpy as np
import cv2
from tqdm import tqdm

## MAKE SURE TO SET THE FOLLOWING FROM CONFIG
# - IMAGE DIMENSIONS (default 224x224)
from config import *

## SET THIS ****
N_FRAMES = 10 
TOP_N = 100
## sets the correct path to read datafiles from 
VIP = True if N_GPUS > 1 else False
DESKTOP = True if N_GPUS == 1 else False
if VIP: 
    PATH2VIDEOS = '/home/brennan/672/videos/'
    DATAFILE_PATH = '/home/brennan/672/regularization_project/top-k-glosses/{}/'.format(TOP_N)
elif DESKTOP: 
    PATH2VIDEOS = '/home/brennan/Documents/gradschool/672/videos/'
    DATAFILE_PATH = '/home/brennan/Documents/gradschool/672/regularization_project/top-k-glosses/{}/'.format(TOP_N)
else: 
    PATH2VIDEOS = '/Users/brennangebotys/Documents/gradschool/672/project/data/WLASL/start_kit/videos/'
    DATAFILE_PATH = '/Users/brennangebotys/Documents/gradschool/672/project/regularization_project/top-k-glosses/{}/'.format(TOP_N)

# %%
##### video reading functions 
def mp4_2_numpy(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    data = np.stack(list(vid.iter_data()))
    return data 

###### tf records functions 
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def line2example(line):
    d = line.split(' ')
    fn, label = '{}{}'.format(PATH2VIDEOS, d[0]), int(d[1])
    
    # read file for shape 
    data = mp4_2_numpy(fn)
    # sample 10 random frames  -- fps == 24 
    t, h, w, c = data.shape
    sampled_frame_idxs = np.linspace(0, t-1, num=N_FRAMES, dtype=np.int32)
    data = data[sampled_frame_idxs]
    t, h, w, c = data.shape

    # save video as list of encoded frames 
    img_bytes = [tf.image.encode_jpeg(d, format='rgb') for d in data]

    with tf.Session() as sess: 
        img_bytes = sess.run(img_bytes)
    
    img_feats = [tf.train.Feature(bytes_list=tf.train.BytesList(value=[imgb])) for imgb in img_bytes]

    sequence_dict = {}
    sequence_dict['data'] = tf.train.FeatureList(feature=img_feats)

    context_dict = {}
    context_dict['filename'] = _bytes_feature(fn.encode('utf-8'))
    context_dict['label'] = _int64_feature(label)
    context_dict['temporal'] = _int64_feature(t)
    context_dict['height'] = _int64_feature(h)
    context_dict['width'] = _int64_feature(w)
    context_dict['depth'] = _int64_feature(c)

    sequence_context = tf.train.Features(feature=context_dict)
    sequence_list = tf.train.FeatureLists(feature_list=sequence_dict)

    example = tf.train.SequenceExample(context=sequence_context, feature_lists=sequence_list)

    return example

#%%    
def create_tfrecords():
    RECORDS_SAVE_PATH = pathlib.Path('data/top-{}/'.format(TOP_N))
    RECORDS_SAVE_PATH.mkdir(exist_ok=True, parents=True)

    for dset_name in ['train.txt', 'test.txt', 'val.txt']:
        with open(DATAFILE_PATH + dset_name, 'r') as f:
            lines = f.readlines()

        record_file = str(RECORDS_SAVE_PATH/'{}.tfrecord'.format(dset_name[:-4]))
        with tf.python_io.TFRecordWriter(record_file) as writer: 
            for line in tqdm(lines[:3]): 
                example = line2example(line)
                writer.write(example.SerializeToString())

#%%
################ READING THE DATA EXAMPLE ####################
def test():
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

        # # sample across temporal frames 
        # idxs = tf.random.uniform((N_FRAMES,), minval=0,
        #         maxval=tf.cast(context['temporal'], tf.int32), 
        #         dtype=tf.int32)
        # video3D = tf.cast(tf.gather(video3D, idxs), tf.float32)

        # resize images in video
        video3D = tf.map_fn(resize, video3D, back_prop=False, parallel_iterations=10)

        label = context['label']

        return video3D, label, context['filename'], shape

    dataset = tf.data.TFRecordDataset('data/top-100/train.tfrecord')\
        .map(_parse_image_function)\
        .batch(2)

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # %%
    with tf.Session() as sess: 
        sess.run([tf.compat.v1.global_variables_initializer(), \
                tf.compat.v1.local_variables_initializer()])
        for _ in range(1):
            vid, _, _, shape = sess.run(next_element)
            print(vid.shape)

if __name__ == "__main__":
    create_tfrecords()     
    test()

