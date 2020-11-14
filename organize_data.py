#%%
import pathlib 
import json
import imageio 
import numpy as np 
from tqdm import tqdm

data_path = pathlib.Path.home()/'Documents/gradschool/672/project/data/WLASL/start_kit/videos/'
filenames = list(data_path.iterdir())

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
