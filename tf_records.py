#%%
import tensorflow as tf 
import pathlib 
import json
import imageio 
import numpy as np 
import cv2 

## MAKE SURE TO SET THE FOLLOWING FROM CONFIG
# - IMAGE DIMENSIONS 
from config import * 

## SET THIS **** 
TOP_N = 100
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

def fn_preprocess(fn):
    vid = mp4_2_numpy(fn)
    t, h, w, _ = vid.shape 

    if h < IMAGE_SIZE_H or w < IMAGE_SIZE_W: 
        vid = np.stack([cv2.resize(frame, (IMAGE_SIZE_W, IMAGE_SIZE_H)) for frame in vid])
    else: 
        # crop to IMAGE_SIZE x IMAGE_SIZE
        h_start = (h - IMAGE_SIZE_H) // 2
        w_start = (w - IMAGE_SIZE_W) // 2
        vid = vid[:, h_start: h_start + IMAGE_SIZE_H, w_start: w_start + IMAGE_SIZE_W, :]
    return vid

###### tf records functions 
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def line2example(line):
    """Takes a dataset line to a mp4 file and returns a tfrecords example.

    Args:
        line ([str]): in the format "{filename} {label}"
    """
    d = line.split(' ')
    fn, label = '{}{}'.format(PATH2VIDEOS, d[0]), int(d[1])

    ## dont pre-process (can fix this later if its better)
    data = mp4_2_numpy(fn)

    t, h, w, c = data.shape
    feature = {
        'filename': _bytes_feature(fn.encode('utf-8')),
        'data': _bytes_feature(data.tostring()),
        'label': _int64_feature(label),
        'temporal': _int64_feature(t),
        'height': _int64_feature(h),
        'width': _int64_feature(w),
        'depth': _int64_feature(c),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

#%%    
RECORDS_SAVE_PATH = pathlib.Path('data/top-{}/'.format(top-k-glosses))
RECORDS_SAVE_PATH.mkdir(exist_ok=True, parents=True)

for dset_name in ['train.txt', 'test.txt', 'val.txt']:
    with open(DATAFILE_PATH + dset_name, 'r') as f: 
        lines = f.readlines()

    with tf.python_io.TFRecordWriter(RECORDS_SAVE_PATH/'{}.tfrecord'.format(dset_name[:-4])) as writer: 
        for line in lines: 
            example = line2example(line)
            writer.write(example.SerializeToString())

#%%
################ READING THE DATA EXAMPLE ####################

feature_description = {
    'filename': tf.io.FixedLenFeature([], tf.string),
    'data': tf.io.FixedLenFeature([], tf.string),
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
    metadata = tf.io.parse_single_example(example_proto, feature_description)
    
    shape = (metadata['temporal'], metadata['height'], metadata['width'], metadata['depth'])
    video3D_buffer = tf.reshape(metadata['data'], shape=[])
    video3D = tf.decode_raw(video3D_buffer, tf.uint8)
    video3D = tf.reshape(video3D, shape)

    # sample across temporal frames 
    n_frames = 10 
    idxs = tf.random.uniform(
            (n_frames,),
            minval=0,
            maxval=tf.cast(metadata['temporal'], tf.int32)-1,
            dtype=tf.int32)
    video3D = tf.cast(tf.gather(video3D, idxs), tf.float32)

    video3D = tf.map_fn(resize, video3D, back_prop=False, parallel_iterations=10)

    label = metadata['label']

    return video3D, label, metadata['filename']

#%%
dataset = tf.data.TFRecordDataset('train.tfrecord')\
    .map(_parse_image_function)\
    .batch(3)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess: 
    while True: 
        e = sess.run(next_element)
        print(e)