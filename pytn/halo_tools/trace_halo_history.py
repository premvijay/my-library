import numpy as np
import pandas as pd
# import sys
import os
# import copy
# import pdb
import h5py
import requests
from PIL import Image
from io import BytesIO



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



def crawl_illustris(halo_id, simname = 'TNG100-1', snapnum_start=98, snapnum_trace=49, snapnum_return='all_across', return_sublink_ind=True, return_R200c=False, return_M200c=False, return_sbhlID=False, return_pos=False, return_satl_info=False):
    halo_id = np.asarray(halo_id)
    res_extras = {}
    if len(halo_id.shape)==0:
        halo_id=halo_id[None]
    
    crawl_len = snapnum_start-snapnum_trace
    with h5py.File(os.environ['SCRATCH'] + f'/download/IllTNG/{simname}/simulation.hdf5', mode='r') as simfile:
        subhalo_id = simfile[f'/Groups/{snapnum_start:d}/Group/GroupFirstSub'][:int(np.max(halo_id)+1)][halo_id]
        sublink_ind = simfile['Offsets/98/Subhalo/SubLink']['RowNum'][:int(np.max(subhalo_id)+1)][subhalo_id]
        sublink_prelod_len = int(np.max(sublink_ind)+crawl_len+1)
        subln_preload_snapnum = simfile['Trees/SubLink']['SnapNum'][:sublink_prelod_len]
        subln_preload_hosthlID = simfile['Trees/SubLink']['SubhaloGrNr'][:sublink_prelod_len]

        # filter_matchhals_ind = np.where((sublink_ind!=-1) & (subln_preload_snapnum[sublink_ind+crawl_len]==snapnum_trace))
        sublink_ind_traced_all = np.linspace(sublink_ind,sublink_ind+crawl_len, crawl_len+1, dtype='int').T

        if snapnum_return=='all_across':
            sublink_ind_traced = sublink_ind_traced_all
        else:
            crawl_len_ret = snapnum_start-snapnum_return
            sublink_ind_traced = sublink_ind+crawl_len_ret
        
        filter_insublink = sublink_ind!=-1
        filter_snapnum =  subln_preload_snapnum[sublink_ind+crawl_len]==snapnum_trace
        # print(filter_snapnum.sum())
        filter_snapnum = ~( subln_preload_snapnum[sublink_ind_traced_all]-np.arange(98,snapnum_trace-1,-1) ).astype(bool).any(axis=1)
        # print(filter_snapnum.sum())

        filter_matchhals_ind = np.where(filter_insublink & filter_snapnum) # & filter_centrals)
        sublink_ind_traced_filt = sublink_ind_traced[filter_matchhals_ind]
        
        halo_id_traced = subln_preload_hosthlID[sublink_ind_traced_filt]
        
        res_list = [halo_id_traced, filter_matchhals_ind]
        if return_sublink_ind:
            res_extras['sbln_ind'] = sublink_ind_traced_filt
        if return_R200c:
            subln_preload_hosthlR200c = simfile['Trees/SubLink']['Group_R_Crit200'][:sublink_prelod_len]
            res_extras['hal_R200c'] = subln_preload_hosthlR200c[sublink_ind_traced_filt]
        if return_M200c:
            subln_preload_hosthlM200c = simfile['Trees/SubLink']['Group_M_Crit200'][:sublink_prelod_len]
            res_extras['hal_M200c'] = subln_preload_hosthlM200c[sublink_ind_traced_filt]

        if return_sbhlID or return_satl_info:
            subln_preload_sbhlID = simfile['Trees/SubLink']['SubfindID'][:sublink_prelod_len]
        
        if return_sbhlID:
            res_extras['sbhlID'] = subln_preload_sbhlID[sublink_ind_traced_filt]

        if return_pos:
            subln_preload_hosthlpos = simfile['Trees/SubLink']['GroupPos'][:sublink_prelod_len]
            res_extras['hal_pos'] = subln_preload_hosthlpos[sublink_ind_traced_filt]
        
        if return_satl_info:
            subln_preload_censbhlID = simfile['Trees/SubLink']['GroupFirstSub'][:sublink_prelod_len]
            res_extras['filter_centrals'] = subln_preload_censbhlID[sublink_ind_traced_filt]==subln_preload_sbhlID[sublink_ind_traced_filt]
            # print(res_extras['filter_centrals'].sum())
            # res_extras['filter_centrals_persist'] = ( subln_preload_censbhlID[sublink_ind_traced_all]==subln_preload_sbhlID[sublink_ind_traced_all] ).all(axis=1)
            # print(res_extras['filter_centrals_persist'].sum())
        
        res_list.append(res_extras)

        # print(filter_matchhals_ind[0].shape)
        # halo_id_start = halo_id[filter_matchhals_ind]
        # halo_id_traced = simfile[f'/Groups/{snapnum_trace:d}/Subhalo/SubhaloGrNr'][:int(subhalo_id_traced.max()+1)][subhalo_id_traced[filter_matchhals_ind]]
    # simfile.close()
    return res_list

# Function to fetch image from URL
def fetch_image(snapnum,hal_id,matter_type,simname='TNG100-1',save=None):
    savefilepath = save+f"_{matter_type}_snap{snapnum}.png"
    if os.path.exists(savefilepath):
        if os.path.getsize(savefilepath)>10000:
            return
    url = f"http://www.tng-project.org/api/{simname}/snapshots/{snapnum}/halos/{hal_id}/vis.png?partType={matter_type}"
    r = requests.get(url, headers={"api-key": "ed0936bfe455b1212682e564fe2324c0"})
    if save==None:
        return Image.open(BytesIO(r.content))
    else:
        with open(savefilepath, "wb") as f:
            f.write(r.content)

def fetch_cutout(snapnum,hal_id,simname='TNG100-1',save=None):
    if save==True:
        save = os.environ['SCRATCH']+f"/download/IllTNG/{simname}/postprocessing/cutouts/"
    savedirpath = save+f"/snap{snapnum}/"
    # print(snapnum,hal_id,simname)
    os.makedirs(savedirpath,exist_ok=True)
    savefilepath = savedirpath+f"/hal{hal_id}.hdf5"
    if os.path.exists(savefilepath):
        if os.path.getsize(savefilepath)>10000:
            return
    # else:
    #     print(snapnum,hal_id,simname, 'not exist')
    #     return
    url = f"http://www.tng-project.org/api/{simname}/snapshots/{snapnum}/halos/{hal_id}/cutout.hdf5"
    # r = requests.get(url, headers={"api-key": "73a59b711c03ef63a039a02fad028d52"})
    consize = 0
    numtry=0
    while consize<10000 and numtry<=3:
        numtry+=1
        try:
            r = requests.get(url, headers={"api-key": "ed0936bfe455b1212682e564fe2324c0"})
            consize = len(r.content)
        except:
            continue # print('error occured doing again',snapnum,hal_id)
    if numtry==4:
        print('error occured',snapnum,hal_id,simname, len(r.content))
    else:
        if save==None:
            return r
        else:
            print(snapnum,hal_id,simname, len(r.content))
            with open(savefilepath, "wb") as f:
                f.write(r.content)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trace most massive progenitor history for a collection of halos, use tree root id to get individual history', usage= 'python')
    parser.add_argument('--halosfile', type=str, help='path of file containing selected halos saved data')
    parser.add_argument('--treesdir', type=str, help='path of directory containing all halos saved data')
    args = parser.parse_args()
    
    mmp_branch(args.halosfile, args.treesdir)
