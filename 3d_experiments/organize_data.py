#%%
import pathlib 
import json
import imageio 
import numpy as np 
import cv2 
from tqdm import tqdm
from config import * 

data_path = pathlib.Path.home()/'Documents/gradschool/672/project/data/WLASL/start_kit/videos/'
filenames = list(data_path.iterdir())
stems = [fn.stem for fn in filenames]

# tf.enable_eager_execution()

#%%
path = pathlib.Path.home()/'Documents/gradschool/672/project/data/pre-processed'
path.mkdir(parents=True, exist_ok=True)

#%%
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

#%%
def preprocess(fn):
    data = fn_preprocess(fn)
    np.save(path/'{}.npy'.format(fn.stem), data)

from tqdm.contrib.concurrent import process_map  
r = process_map(preprocess, filenames, max_workers=10)
    
#%%
def get_metadata():
    file_path = pathlib.Path.home()/'Documents/gradschool/672/project/data/WLASL/start_kit/WLASL_v0.3.json'
    with open(file_path) as ipf:
        metadata = json.load(ipf)
    return metadata

metadata = get_metadata()
len_glosses = [len(m['instances']) for m in metadata]
topk_idxs = np.argsort(len_glosses)[-300:][::-1]

#%%
idxs = topk_idxs
train, val, test = [], [], []
# i also is the label for the class 
for i, idx in tqdm(enumerate(idxs)): 
    classi = metadata[i]
    for ex in classi['instances']:
        split = ex['split']
        id = ex['video_id']
        x = (id, i)
        if split == 'train':
            train.append(x)
        elif split == 'test':
            test.append(x) 
        elif split == 'val': 
            val.append(x)
        else: 
            assert False, "Invalid Split {}".format(split)
    
#%%
basedir = 'top-k-glosses/{}/'.format('300')
for fn, data in zip(['train.txt', 'test.txt', 'val.txt'], [train, test, val]):
    with open(basedir + fn, 'w') as f: 
        for video_id, label in tqdm(data):
            line = '{} {}\n'.format(video_id, label)
            f.write(line)

#%%
def fn2label(fn, include_split=False):
    label = None 
    for i, c in enumerate(metadata): 
        for ex in c['instances']:
            if ex['video_id'] == fn.stem:
                if include_split: 
                    return i, ex['split']
                else: 
                    return i 
    raise 'Label cannot be found for file {}'.format(fn)
                
#%%
from tqdm.notebook import tqdm 
train, test, val = [], [], []
for fn in tqdm(filenames):
    label, split = fn2label(fn, include_split=True)
    # top 100 filter 
    if label in top100_idxs:
        x = (fn, label)
        if split == 'train':
            train.append(x)
        elif split == 'test':
            test.append(x) 
        elif split == 'val': 
            val.append(x)
        else: 
            assert False, "Invalid Split {}".format(split)

# #%%
for fn, data in zip(['train.txt', 'test.txt', 'val.txt'], [train, test, val]):
    with open(fn, 'w') as f: 
        for fn, label in tqdm(data):
            line = '{} {}\n'.format(fn, label)
            f.write(line)

#%%
import imageio 
from tqdm.notebook import tqdm 

## Check to ensure all files are valid 
for dset_file in ['test.txt', 'val.txt']:
    with open(dset_file, 'r') as f: 
        lines = f.readlines()
    for line in tqdm(lines): 
        try: 
            fn = line.split(' ')[0]
            imageio.get_reader(fn, 'ffmpeg')
        except: 
            print('failed: ', fn)

# %%
## Check to ensure all files are valid 
path = 'top-k-glosses/300/'
train, test, val = [], [], []
for dset_file in ['train.txt', 'test.txt', 'val.txt']:
    with open(path+dset_file, 'r') as f: 
        lines = f.readlines()

    for line in tqdm(lines): 
        try: 
            fn = line.split(' ')[0]
            label = line.split(' ')[1]
            if fn[:-4] in stems: 
                x = (fn, label)
                if dset_file == 'train.txt':
                    train.append(x)
                elif dset_file == 'test.txt':
                    test.append(x)
                else: 
                    val.append(x)
        except: 
            print('failed: ', fn)

# %%
for dset_file, data in zip(['train.txt', 'test.txt', 'val.txt'], [train, test, val]):
    with open(path+dset_file, 'w') as f: 
        for fn, label in tqdm(data):
            line = '{} {}'.format(fn, label)
            f.write(line)

###### TF RECORDS ROUGH WORK ####### 

# %%
with open(DATAFILE_PATH+'train.txt', 'r') as f: 
    lines = f.readlines()
    
def mp4_2_numpy(filename):
    vid = imageio.get_reader(filename, 'ffmpeg')
    data = np.stack(list(vid.iter_data()))
    return data 

# %%
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

N_FRAMES = 10 

def line2example(line):
    d = line.split(' ')
    fn, label = '{}{}'.format(PATH2VIDEOS, d[0]), int(d[1])
    
    # read file for shape 
    data = mp4_2_numpy(fn)
    t, h, w, c = data.shape
    # sample 10 random frames  -- fps == 24 
    sampled_frame_idxs = np.linspace(0, t-1, num=N_FRAMES, dtype=np.int32)
    vid = vid[sampled_frame_idxs]

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

with tf.python_io.TFRecordWriter('data/example2.tfrecord') as writer: 
    for line in tqdm(lines[:4]):
        example = line2example(line)
        writer.write(example.SerializeToString())

#%%
tf.reset_default_graph()

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
    video3D = video_data
    # use this to set the shape -- doesn't change anything 
    video3D = tf.reshape(video3D, shape)

    # sample across temporal frames 
    n_frames = 10 
    idxs = tf.random.uniform((n_frames,), minval=0,
            maxval=tf.cast(context['temporal'], tf.int32), 
            dtype=tf.int32)
    video3D = tf.cast(tf.gather(video3D, idxs), tf.float32)

    video3D = tf.map_fn(resize, video3D, back_prop=False, parallel_iterations=10)

    label = context['label']

    return video3D, label, context['filename'], shape

dataset = tf.data.TFRecordDataset('data/example2.tfrecord')\
    .map(_parse_image_function)\
    .batch(3)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# %%
with tf.Session() as sess: 
    sess.run([tf.compat.v1.global_variables_initializer(), \
            tf.compat.v1.local_variables_initializer()])
    for _ in range(1):
        vid, _, _, shape = sess.run(next_element)
        print(vid.shape)

# %%