# print('enter python')
import numpy as np
import h5py

def add_params(filepath):
    f = h5py.File(filepath, 'r+')
    f['Header'].attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype=int) )
    for key in ['HubbleParam', 'Omega0', 'OmegaLambda']:
        f['Header'].attrs.create(key, f['Parameters'].attrs[key])
    f.close()

if __name__=="__main__":
    # print('using python to edit hdf5')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("snapfile", help="snapshot hdf5 full filename")
    args = parser.parse_args()
    add_params(args.snapfile)
    print(args.snapfile)
    