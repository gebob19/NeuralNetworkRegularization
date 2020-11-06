#%%
import pathlib 
data_path = pathlib.Path.home()/'Documents/gradschool/672/project/data/WLASL/start_kit/videos/'
filenames = list(data_path.iterdir())

#%%
import imageio 
def mp4_2_numpy(fn):
    vid = imageio.get_reader(fn, 'ffmpeg')
    data = np.stack(list(vid.iter_data()))
    return data 

#%%
mp4_2_numpy(filenames[0]).shape

#%%