
#%%
from functools import partial
import os
import sys
import pickle
import numpy as np
import pandas as pd
# import warnings
from time import time, sleep
import h5py
from h5my import saveh5_dict


from scipy.special import erf
# from scipy.integrate import quad
from numpy import pi, sqrt, exp
# import numba
# %%
# import pandas as pd
# %%
from scipy.interpolate import interp1d

#%%
from pyread_eagle import EagleSnapshot

#%%
from colossus.cosmology import cosmology
from colossus.halo import concentration

from pm_tools import unperiod


#%%
# def mymeshgrid(x,y):
#     shape = (x.shape[0],y.shape[0])
#     X = np.zeros(shape)
#     Y = np.zeros(shape)
#     X.T[:] = x
#     Y[:] = y
#     return X.T,Y.T

# def meshgrid_2nd(x,y):
#     shape = (y.shape[0],x.shape[0])
#     # X = np.zeros(shape)
#     Y = np.zeros(shape, dtype=np.float32)
#     # X[:] = x
#     Y.T[:] = y
#     return Y
# @numba.jit
def gaussian_integ_sphere_indef_core(r,R):
    # print(r.dtype,R.dtype)
    return ( erf((r - R)/sqrt(2)) + erf((r + R)/sqrt(2)) ) /2 - ( (exp(-1/2 *(r - R)**2) - exp(-1/2 *(r + R)**2)) ) /sqrt(2*pi)/R

def gaussian_integ_sphere_indef(Rad, prtcl_pos_r, h=1, m=1):
    t_now = time()
#     Rad_h = Rad/h
#     pos_r_h = prtcl_pos_r/h
    # R, r = np.meshgrid(prtcl_pos_r, Rad, copy=True, sparse=False)
    # r = meshgrid_2nd(prtcl_pos_r, Rad)
    r = np.zeros((Rad.shape[0],prtcl_pos_r.shape[0]), dtype=np.float32)
    r.T[:] = Rad
    # t_bef, t_now = t_now, time()
    # print(t_now-t_bef, 'meshgrid obtained')
    # R, r = R.squeeze(), r.squeeze()
    # print(r.shape, r.nbytes, r.nbytes)
    h, m = np.asarray(h), np.asarray(m)
#     r =r.T
#     print(r,h)
    r /= h
    R = (prtcl_pos_r/h).astype(np.float32)[np.newaxis,:]
    # t_bef, t_now = t_now, time()
    # print(t_now-t_bef, 'normalised')
    # print('\n And posits', R.shape, R)

    ans = gaussian_integ_sphere_indef_core(r,R)
    # t_bef, t_now = t_now, time()
    # print(t_now-t_bef, 'core computed')
    del r, R
    ans *= m
    # t_bef, t_now = t_now, time()
    # print(t_now-t_bef, 'Weighted by mass')
    return ans

# %%
def read_prtcl_hal_pairs_tng(mtch_pair, simname, savefilepth=None, snapnum=98, useCutout=False):
    halo_id, halo_dmo_id = int(mtch_pair.ID), int(mtch_pair.ID_dmo)
    simfile = h5py.File(os.environ['SCRATCH'] + f'/download/IllTNG/{simname}/simulation.hdf5', mode='r')
    simfile_dmo = h5py.File(os.environ['SCRATCH'] + f'/download/IllTNG/{simname}-Dark/simulation.hdf5', mode='r')
    # print('reading', halo_id)
    box_size = simfile['Header'].attrs['BoxSize']
    grpfil = simfile[f'/Groups/{snapnum}/Group']
    grpfil_dmo = simfile_dmo[f'/Groups/{snapnum}/Group']

    cen = grpfil['GroupPos'][halo_id][:]
    cen_dmo = grpfil_dmo['GroupPos'][halo_dmo_id][:]

    if useCutout:
        cutoutfile = h5py.File(os.environ['SCRATCH']+f"/download/IllTNG/{simname}/postprocessing/cutouts/snap{snapnum}/hal{halo_id}.hdf5", mode='r')
        cutoutfile_dmo = h5py.File(os.environ['SCRATCH']+f"/download/IllTNG/{simname}-Dark/postprocessing/cutouts/snap{snapnum}/hal{halo_dmo_id}.hdf5", mode='r')
            
        posd = unperiod(cutoutfile[f'/PartType1/Coordinates'][:] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)

        posb = unperiod(cutoutfile[f'/PartType0/Coordinates'][:] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)
        m_prtb = cutoutfile[f'/PartType0/Masses'][:]
        densb = cutoutfile[f'/PartType0/Density'][:]
        hsmlb = (m_prtb/densb)**(1/3)

        pos_star = unperiod(cutoutfile[f'/PartType4/Coordinates'][:] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)
        m_prts = cutoutfile[f'/PartType4/Masses'][:]

        if cutoutfile['Header'].attrs['NumPart_ThisFile'][5] > 0:
            pos_bh = unperiod(cutoutfile[f'/PartType5/Coordinates'][:] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)
            m_prtbh = cutoutfile[f'/PartType5/Masses'][:]
        else:
            m_prtbh = None

        posd_dmo = unperiod(cutoutfile_dmo[f'/PartType1/Coordinates'][:] - grpfil_dmo['GroupPos'][halo_dmo_id], lenscl=box_size/2, box_size=box_size)
        
        z = cutoutfile[f'Header'].attrs['Redshift']

        cutoutfile.close()
        cutoutfile_dmo.close()

    else:
        start = simfile[f'/Offsets/{snapnum}/Group/SnapByType'][halo_id, 1]
        length = simfile[f'/Groups/{snapnum}/Group/GroupLenType'][halo_id, 1]
        posd = unperiod(simfile[f'/Snapshots/{snapnum}/PartType1/Coordinates'][start:start+length] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)

        start = simfile[f'/Offsets/{snapnum}/Group/SnapByType'][halo_id, 0]
        length = simfile[f'/Groups/{snapnum}/Group/GroupLenType'][halo_id, 0]
        posb = unperiod(simfile[f'/Snapshots/{snapnum}/PartType0/Coordinates'][start:start+length] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)
        m_prtb = simfile[f'/Snapshots/{snapnum}/PartType0/Masses'][start:start+length]
        densb = simfile[f'/Snapshots/{snapnum}/PartType0/Density'][start:start+length]
        hsmlb = (m_prtb/densb)**(1/3)

        start = simfile[f'/Offsets/{snapnum}/Group/SnapByType'][halo_id, 4]
        length = simfile[f'/Groups/{snapnum}/Group/GroupLenType'][halo_id, 4]
        pos_star = unperiod(simfile[f'/Snapshots/{snapnum}/PartType4/Coordinates'][start:start+length] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)
        m_prts = simfile[f'/Snapshots/{snapnum}/PartType4/Masses'][start:start+length]

        start = simfile[f'/Offsets/{snapnum}/Group/SnapByType'][halo_id, 5]
        length = simfile[f'/Groups/{snapnum}/Group/GroupLenType'][halo_id, 5]
        pos_bh = unperiod(simfile[f'/Snapshots/{snapnum}/PartType5/Coordinates'][start:start+length] - grpfil['GroupPos'][halo_id], lenscl=box_size/2, box_size=box_size)
        m_prtbh = simfile[f'/Snapshots/{snapnum}/PartType5/Masses'][start:start+length]

        start_dmo = simfile_dmo[f'/Offsets/{snapnum}/Group/SnapByType'][halo_dmo_id, 1]
        length_dmo = simfile_dmo[f'/Groups/{snapnum}/Group/GroupLenType'][halo_dmo_id, 1]
        posd_dmo = unperiod(simfile_dmo[f'/Snapshots/{snapnum}/PartType1/Coordinates'][start_dmo:start_dmo+length_dmo] - grpfil_dmo['GroupPos'][halo_dmo_id], lenscl=box_size/2, box_size=box_size)

        z = simfile[f'Snapshots/{snapnum}/Header'].attrs['Redshift']
    
    if m_prtbh is None:
        pos_starbh = pos_star
        m_prt_starbh = m_prts
    else:
        pos_starbh = np.concatenate([pos_star,pos_bh])
        m_prt_starbh = np.concatenate([m_prts,m_prtbh])

    m_prtd = simfile['Header'].attrs['MassTable'][1]
    m_prtd_dmo = simfile_dmo['Header'].attrs['MassTable'][1]
    Rvir = simfile[f'/Groups/{snapnum}/Group/Group_R_Crit200'][halo_id]
    Rvir_dmo = simfile_dmo[f'/Groups/{snapnum}/Group/Group_R_Crit200'][halo_dmo_id]
    f_dm = 1 - simfile['Header'].attrs['OmegaBaryon']/simfile['Header'].attrs['Omega0']

    eps_sl = simfile['Parameters'].attrs['SofteningMaxPhysType1'] * (1+z)

    data = {'cen':cen, 'cen_dmo':cen_dmo, 'posd_dmo':posd_dmo, 'posd':posd, 'posb':posb, 'pos_star':pos_starbh, 'm_prtd_dmo':m_prtd_dmo*1e10, 'm_prtd':m_prtd*1e10, 'm_prtb':m_prtb*1e10, 'm_prt_star':m_prt_starbh*1e10, 'hsmlb':hsmlb, 'fd':f_dm, 'Rvir':Rvir, 'Rvir_dmo':Rvir_dmo, 'ID':halo_id, 'ID_dmo':halo_dmo_id, 'eps_sl':eps_sl }
    # print('saving')
    simfile.close()
    simfile_dmo.close()
    if savefilepth==None:
        return data
    else:
        saveh5_dict(savefilepth + f"/preread/{simname}_{halo_id:d}.hdf5", data, comp='gzip')

from itertools import chain, combinations
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss)+1)))

def read_prtcl_hal_pairs_eagle(mtch_pair, simfilename, simfilename_dmo, savefilepth=None, snapnum=28):
    snap = EagleSnapshot(simfilename)
    snap_dmo = EagleSnapshot(simfilename_dmo)
    snaph5 = h5py.File(simfilename,'r'); snaph5_dmo = h5py.File(simfilename_dmo,'r')
    box_size = snaph5['Header'].attrs['BoxSize']
    fd = 1 - snaph5['Header'].attrs['OmegaBaryon']/snaph5['Header'].attrs['Omega0']
    m_prtd = snaph5['Header'].attrs['MassTable'][1]*1e10
    m_prtd_dmo = snaph5_dmo['Header'].attrs['MassTable'][1]*1e10
    eps_sl = snaph5['RuntimePars'].attrs['SofteningHaloMaxPhys']*1e3
    snaph5.close(); snaph5_dmo.close()
    halo_id, halo_dmo_id = int(mtch_pair.ID), int(mtch_pair.ID_dmo)
    cen = mtch_pair[['X','Y','Z']].to_numpy()/1e3
    Rvir = mtch_pair.R/1e3
    reg = np.zeros(6)
    reg[::2] = cen - 1.2*Rvir
    reg[1::2] = cen + 1.2*Rvir
    snap.clear_selection()
    snap.select_region(*reg)

    cen_dmo = mtch_pair[['Xd','Yd','Zd']].to_numpy()/1e3
    Rvir_dmo = mtch_pair.R_dmo/1e3
    reg_dmo = np.zeros(6)
    reg_dmo[::2] = cen_dmo - 1.2*Rvir_dmo
    reg_dmo[1::2] = cen_dmo + 1.2*Rvir_dmo
    snap_dmo.clear_selection()
    snap_dmo.select_region(*reg_dmo)

    # for d in range(3):
    #     if reg_dmo[::2][d]<0:
    #         reg_dmo[::2][d] +=25
    #         reg_dmo[1::2][d] +=25
    #         snap_dmo.select_region(*reg_dmo)
    if np.any(reg<0) or np.any(reg_dmo<0):
        bound_hit = np.where(reg<0)
        for subset in all_subsets(bound_hit):
            for d in subset:
                reg[d] += box_size
                reg[d+1] += box_size
            snap.select_region(*reg)
        bound_hit = np.where(reg_dmo<0)
        for subset in all_subsets(bound_hit):
            for d in subset:
                reg_dmo[d] += box_size
                reg_dmo[d+1] += box_size
            snap_dmo.select_region(*reg_dmo)

    gnsd_dmo = snap_dmo.read_dataset(1, 'GroupNumber')
    posd_dmo = unperiod(snap_dmo.read_dataset(1, "Coordinates")[gnsd_dmo==halo_dmo_id] - cen_dmo, lenscl=box_size/2, box_size=box_size)

    gnsd = snap.read_dataset(1, 'GroupNumber')
    posd = unperiod(snap.read_dataset(1, "Coordinates")[gnsd==halo_id] - cen, lenscl=box_size/2, box_size=box_size)

    gnsb = snap.read_dataset(0, 'GroupNumber')
    indxb = np.where(gnsb==halo_id)
    posb = unperiod(snap.read_dataset(0, "Coordinates")[indxb] - cen, lenscl=box_size/2, box_size=box_size)
    m_prtb = snap.read_dataset(0, "Mass")[indxb]
    hsmlb = snap.read_dataset(0, "SmoothingLength")[indxb]

    gns_star = snap.read_dataset(4, 'GroupNumber')
    indx_star = np.where(gns_star==halo_id)
    pos_star = unperiod(snap.read_dataset(4, "Coordinates")[indx_star] - cen, lenscl=box_size/2, box_size=box_size)
    m_prts = snap.read_dataset(4, "Mass")[indx_star]

    gns_bh = snap.read_dataset(5, 'GroupNumber')
    indx_bh = np.where(gns_bh==halo_id)
    try:
        pos_bh = snap.read_dataset(5, "Coordinates")[indx_bh] - cen
        m_prtbh = snap.read_dataset(5, "Mass")[indx_bh]
    
        pos_starbh = np.concatenate([pos_star,pos_bh])
        m_prt_starbh = np.concatenate([m_prts,m_prtbh])
    except:
        pos_starbh = pos_star
        m_prt_starbh = m_prts

    snap.close()
    snap_dmo.close()

    data = {'cen':cen*1e3, 'cen_dmo':cen_dmo*1e3, 'posd_dmo':posd_dmo*1e3, 'posd':posd*1e3, 'posb':posb*1e3, 'pos_star':pos_starbh*1e3, 'm_prtb':m_prtb*1e10, 'm_prt_star':m_prt_starbh*1e10, 'hsmlb':hsmlb, 'Rvir':Rvir*1e3, 'Rvir_dmo':Rvir_dmo*1e3, 'ID':halo_id, 'ID_dmo':halo_dmo_id, 'fd':fd, 'm_prtd':m_prtd, 'm_prtd_dmo':m_prtd_dmo, 'eps_sl':eps_sl }
    # print('saving')
    if savefilepth==None:
        return data
    else:
        raise NotImplementedError
        # saveh5_dict(savefilepth + f"/preread/{simname}_{halo_id:d}_{halo_dmo_id:d}.hdf5", data, comp=None)#'gzip')
        # np.save(savefilepth, data)


#%%
def read_prtcl_hal_pairs_camels(mtch_pair, simname, simset0='I', snapnum=33):
    halo_id, halo_dmo_id = int(mtch_pair.ID), int(mtch_pair.ID_dmo)
    halname = f"{simname}_snap0{snapnum}_{halo_id:d}_{halo_dmo_id:d}"
    return h5py.File(os.environ['SCRATCHLOCAL']+f"/halo_data/preread/CAMELS_{simset0}/1P/{halname}.hdf5", 'r')
    

#%%
def get_rel_ratio_conveni_wrap(args):
    arg_dict, mtch_pair = args
    range_min_r, eps_sl, fd, m_prtd, m_prtd_dmo, sph_gas, rbins_num= arg_dict['common']
    # halo_id, halo_dmo_id = mtch_pair.ID, mtch_pair.ID_dmo
    # halname = f"{simname}_{halo_id:d}_{halo_dmo_id:d}"
    # data = np.load(os.environ['SCRATCHLOCAL']+f"/tmp/{halname}.npy", allow_pickle=True).tolist()
    # data = h5py.File(os.environ['SCRATCHLOCAL']+f"/halo_data/preread/{halname}.hdf5", 'r')
    # data_attrs = data.attrs
    # print(mtch_pair)

    savedir = os.environ['SCRATCHLOCAL']+f"/adiab-relx/cached/{arg_dict['simsuite']}/"
    os.makedirs(savedir, exist_ok=True)
    savepath=savedir+f"{arg_dict['simname']}_{arg_dict['snapnum']}_{int(mtch_pair.ID)}_{int(mtch_pair.ID_dmo)}_{rbins_num}.hdf5"
    if os.path.exists(savepath):
        # try:
            with h5py.File(savepath) as cached_hf:
                res = cached_hf
                return (res['MiMf'][:], res['rfri'][:], res['rf_by_R'][:], res['Mdr'][:], res['Mbr'][:], res['Msr'][:], res['Mdr_dmo'][:], res['ri_pre_by_R'][:])
        # except:
            # pass
            # print('exception',savepath)
    data_attrs = None
    if arg_dict['simsuite']=='Eagle':
        data = read_prtcl_hal_pairs_eagle(mtch_pair, arg_dict['simfilename'], arg_dict['simfilename_dmo'], snapnum=arg_dict['snapnum'])
    elif arg_dict['simsuite']=='Tng':
        try:
            data = read_prtcl_hal_pairs_tng(mtch_pair, simname=arg_dict['simname'], snapnum=arg_dict['snapnum'], useCutout=arg_dict['useCutout'])
        except:
            return [np.full((rbins_num,),np.nan),]*6+[np.full((rbins_num*2,),0),]*2
    elif arg_dict['simsuite']=='Camels':
        data = read_prtcl_hal_pairs_camels(mtch_pair, simname=arg_dict['simname'], simset0=arg_dict['simset'][0], snapnum=arg_dict['snapnum'])
        data_attrs = data.attrs
    
    res = get_rel_ratio(data, data_attrs, sph_gas=sph_gas, rbins_num=rbins_num, range_min_r=range_min_r, warn_noise=arg_dict['warn_noise'], noise_setnan=arg_dict['noise_setnan'], noDicres=0)
    saveh5_dict(savepath, res)
    return (res['MiMf'][:], res['rfri'][:], res['rf_by_R'][:], res['Mdr'][:], res['Mbr'][:], res['Msr'][:], res['Mdr_dmo'][:], res['ri_pre_by_R'][:])
    
    
def get_rel_ratio(data, data_attrs=None, sph_gas=1, rbins_num=30, range_min_r=None, warn_noise=0, noise_setnan=0, relx_prtcl=1, range_max_by_Rvir=1, noDicres=1):
    noisy=0
    if data_attrs==None: data_attrs = data
    eps_sl, fd, m_prtd, m_prtd_dmo = data_attrs['eps_sl'], data_attrs['fd'], data_attrs['m_prtd'], data_attrs['m_prtd_dmo']
    R = data_attrs['Rvir']
    R_dmo = data_attrs['Rvir_dmo']
    if range_min_r==None: range_min_r = 10 * eps_sl / R
    range_min_dmo = 9 * eps_sl
    range_max = R*range_max_by_Rvir #*.4
    range_max_dmo = R_dmo
    # R = data_attrs['Rvir']
    range_min = range_min_r*R

    Rad_bin_edge = np.logspace(np.log10(range_min),np.log10(range_max), rbins_num)
    Rad_bin_edge_i = np.logspace(np.log10(range_min_dmo),np.log10(range_max_dmo), rbins_num*2)

    Rad_bin_edge = np.insert(Rad_bin_edge,0,0)
    Rad_bin_edge.sort()
    Rad_bin_edge_i = np.insert(Rad_bin_edge_i,0,0)
    Rad_bin_edge_i.sort()
    # Rad_bin_edge
    

    Rad_bin_cen = (Rad_bin_edge[1:] * Rad_bin_edge[:-1])**(1/2)

    posd_r = np.linalg.norm(data['posd'], axis=1)
    num_profile = np.histogram(posd_r, Rad_bin_edge)[0]
    # print(Rad_bin_edge, num_profile)

    

    # assert num_profile.min()>50, f" Noise: Only {num_profile.min():d} particles in a bin, increase bin width. Id: {data_attrs['ID']:d}, {num_profile.argmin():d}"
    if num_profile.min()<20 and warn_noise: 
        noisy=1
        print(f"Warning: Only {num_profile.min():d} particles in a bin, increase bin width. Id: {data_attrs['ID']:d}, {num_profile.argmin():d}")

    mass_profile = num_profile * m_prtd


    posb_r = np.linalg.norm(data['posb'], axis=1)
    hsmlb = data['hsmlb'] #
    massb = data['m_prtb']

    if sph_gas:
        mass_profile_cumm_indv = gaussian_integ_sphere_indef(Rad_bin_edge, posb_r, h=hsmlb, m=massb)
        mass_profile_cumm_gas = np.sum(mass_profile_cumm_indv, dtype=np.float64, axis=1)
        del mass_profile_cumm_indv
        mass_profile_gas = np.diff(mass_profile_cumm_gas)
    else:
        mass_profile_gas = np.histogram(posb_r, Rad_bin_edge, weights=data['m_prtb'])[0]

    pos_star_r = np.linalg.norm(data['pos_star'], axis=1)

    if data['m_prt_star'].size==pos_star_r.size:
        mass_profile_star = np.histogram(pos_star_r, Rad_bin_edge, weights=data['m_prt_star'])[0]
    else:
        mass_profile_star = np.histogram(pos_star_r, Rad_bin_edge)[0]* data['m_prt_star'][0]

    mass_profile_bar = mass_profile_gas + mass_profile_star


    posd_dmo_r = np.linalg.norm(data['posd_dmo'], axis=1)

    if not relx_prtcl:
        num_profile_dmo = np.histogram(posd_dmo_r, Rad_bin_edge_i)[0]

        # print(range_min_dmo, range_max_dmo, posd_dmo_r.max(), posd_dmo_r.min())
        # print(num_profile_dmo)

        # assert num_profile_dmo.min()>20, f"Noise: Only {num_profile_dmo.min():d} particles in a bin, increase DMO bin width. Id_dmo: {data_attrs['ID_dmo']:d}, {num_profile_dmo.argmin():d} \n {num_profile_dmo} {Rad_bin_edge_i}"
        if num_profile_dmo.min()<20 and warn_noise:
            noisy=1
            print(f"Warning: Only {num_profile_dmo.min():d} particles in a bin, increase DMO bin width. Id_dmo: {data_attrs['ID_dmo']:d}, {num_profile_dmo.argmin():d}")

        mass_profile_dmo = num_profile_dmo * m_prtd_dmo

    try:
        data.close() 
    except:
        pass


    r, Mdr, Mbr, Msr = Rad_bin_edge[1:], np.cumsum(mass_profile), np.cumsum(mass_profile_bar), np.cumsum(mass_profile_star)
    ri_pre = Rad_bin_edge_i[1:]
    # r, Mdr, Mbr, Mdr_dmo

    rf = r.copy()

    if relx_prtcl:
        posd_r = np.sort(posd_r)
        # posb_r_sort_ind = np.argsort(posb_r)
        # posb_r = posb_r[posb_r_sort_ind]
        # posb_r = posb_r[:][posb_r_sort_ind]
        # hsmlb = hsmlb[:][posb_r_sort_ind]
        # massb = massb[:][posb_r_sort_ind]
    
        posd_dmo_r = np.sort(posd_dmo_r)

        Ndr = np.searchsorted(posd_r, rf)
        try:
            ri = posd_dmo_r[Ndr-1]
        except:
            ri = 0*Ndr
        # Nbr = np.searchsorted(posb_r, rf)

        # MiMf = ( fd* (Nbr/Ndr *0.049/0.3 + 1) )**-1
        Mdr_dmo = ri_pre*np.nan

    else:
        Mdr_dmo = np.cumsum(mass_profile_dmo)*fd
        logri_logM = interp1d(np.log10(Mdr_dmo),np.log10(ri_pre), fill_value='extrapolate')

        # assert (ri_M(Mdr_dmo) == r).all()

        ri = 10**logri_logM(np.log10(Mdr))

    # Mf = Mdr+Mbr
    # Mi = Mdr/fd

    MiMf = ( fd* (Mbr/ Mdr + 1) )**-1
    rfri = rf / ri

    if noise_setnan and noisy: rfri[:] = np.nan

    if noDicres:
        return (MiMf, rfri, rf/R, Mdr, Mbr, Msr, Mdr_dmo, ri_pre/R)
    
    resdict = {'MiMf':MiMf, 'rfri':rfri, 'rf_by_R':rf/R, 'Mdr':Mdr, 'Mbr':Mbr, 'Msr':Msr, 'Mdr_dmo':Mdr_dmo, 'ri_pre_by_R':ri_pre/R}
    return resdict