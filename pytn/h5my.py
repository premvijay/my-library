import numpy as np
import h5py

def saveh5_dict(filename, datadict={}, comp="lzf", comp_level=1):
    with h5py.File(filename, 'w') as h5fil:
        for k, v in datadict.items():
            if np.isscalar(v):
                h5fil.attrs.create(k, data=v)
            else:
                h5fil.create_dataset(k, data=v, compression=comp) #, compression_opts=comp_level)

