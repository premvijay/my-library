import numpy as np
import pandas as pd
# import sys
import os
# import copy
# import pdb
import h5py



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



def crawl_illustris(halo_id, simname = 'TNG100-1', snapnum_start=98, snapnum_trace=49, all_across=False, return_R200c=True):
    crawl_len = snapnum_start-snapnum_trace
    with h5py.File(os.environ['SCRATCH'] + f'/download/IllTNG/{simname}/simulation.hdf5', mode='r') as simfile:
        subhalo_id = simfile[f'/Groups/{snapnum_start:d}/Group/GroupFirstSub'][:int(np.max(halo_id)+1)][halo_id]
        sublink_ind = simfile['Offsets/98/Subhalo/SubLink']['RowNum'][:int(np.max(subhalo_id)+1)][subhalo_id]
        sublink_prelod_len = int(np.max(sublink_ind)+crawl_len+1)
        subln_preload_snapnum = simfile['Trees/SubLink']['SnapNum'][:sublink_prelod_len]
        # subln_preload_sbhlID = simfile['Trees/SubLink']['SubfindID'][:sublink_prelod_len]
        subln_preload_hosthlID = simfile['Trees/SubLink']['SubhaloGrNr'][:sublink_prelod_len]

        filter_matchhals_ind = np.where((sublink_ind!=-1) & (subln_preload_snapnum[sublink_ind+crawl_len]==snapnum_trace))

        # subhalo_id_traced = subln_preload_sbhlID[sublink_ind+crawl_len]
        if all_across:
            halo_id_traced = subln_preload_hosthlID[np.linspace(sublink_ind,sublink_ind+crawl_len, crawl_len+1, dtype='int')].T
        else:
            halo_id_traced = subln_preload_hosthlID[sublink_ind+crawl_len]
        
        res_list = [halo_id_traced[filter_matchhals_ind], filter_matchhals_ind]
        if return_R200c:
            subln_preload_hosthlR200c = simfile['Trees/SubLink']['Group_R_Crit200'][:sublink_prelod_len]
            halo_R200c_traced = subln_preload_hosthlR200c[sublink_ind+crawl_len]
            res_list.append(halo_R200c_traced[filter_matchhals_ind])

        # print(filter_matchhals_ind[0].shape)
        # halo_id_start = halo_id[filter_matchhals_ind]
        # halo_id_traced = simfile[f'/Groups/{snapnum_trace:d}/Subhalo/SubhaloGrNr'][:int(subhalo_id_traced.max()+1)][subhalo_id_traced[filter_matchhals_ind]]
    # simfile.close()
    return res_list


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trace most massive progenitor history for a collection of halos, use tree root id to get individual history', usage= 'python')
    parser.add_argument('--halosfile', type=str, help='path of file containing selected halos saved data')
    parser.add_argument('--treesdir', type=str, help='path of directory containing all halos saved data')
    args = parser.parse_args()
    
    mmp_branch(args.halosfile, args.treesdir)
