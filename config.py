IMAGE_SIZE_H, IMAGE_SIZE_W = 224, 224
BATCH_SIZE = 8
PREFETCH_BUFFER = BATCH_SIZE
NUM_CLASSES = 100

import tensorflow as tf 
N_GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
VIP = True if N_GPUS > 1 else False 

if VIP: 
    PATH2VIDEOS = '/home/brennan/672/videos/'
    DATAFILE_PATH = '/home/brennan/672/regularization_project/top-k-glosses/100/'
else: 
    PATH2VIDEOS = '/Users/brennangebotys/Documents/gradschool/672/project/data/WLASL/start_kit/videos/'
    DATAFILE_PATH = '/Users/brennangebotys/Documents/gradschool/672/project/regularization_project/top-k-glosses/100/'