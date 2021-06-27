import numpy as np
import pandas as pd
import sys
import os
import copy
import pdb



def is_unique(ss):
    a = ss.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()


def mmp_branch(halosfile, treesdir, upto=1):
    halos_select = pd.read_csv(halosfile, sep=',', engine='c')
    halos_select.set_index('Depth_first_ID(28)', inplace=True)
    if is_unique(halos_select['Snap_num(31)']):
        i = halos_select['Snap_num(31)'].iloc[0]
    else:
        raise Exception('All halos in the collection must be at same redshift')


    while i>int(upto):
        i -= 1
        treefile = os.path.join(treesdir, 'out_{0:d}.trees'.format(i))
        if not os.path.exists(treefile):
            print('leaf reached')
            break
        else:
            print(treefile)
        
        halos = pd.read_csv(treefile, sep=r'\s+', header=0, skiprows=list(range(1,58)), engine='c')#usecols = [0,1,2])
        halos = halos[halos['pid(5)']==-1]
        halos.set_index('Depth_first_ID(28)', inplace=True)

        # halos_select_previous = copy.deepcopy(halos_select)
        halos_to_look = halos_select.index + 1
        # print(list(halos.columns))
        # pdb.set_trace()

        halos_to_select = []
        for Depth_ID in halos_to_look:
            if Depth_ID in halos.index:
                halos_to_select.append(Depth_ID)

        halos_select = halos.loc[halos_to_select]

        halos_select['Snap_num(31)'] = int(i)
        halos_select.to_csv(halosfile, mode='a', header=False)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trace most massive progenitor history for a collection of halos, use tree root id to get individual history', usage= 'python')
    parser.add_argument('--halosfile', type=str, help='path of file containing selected halos saved data')
    parser.add_argument('--treesdir', type=str, help='path of directory containing all halos saved data')
    args = parser.parse_args()
    
    mmp_branch(args.halosfile, args.treesdir)
