import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree as KDTree_skl
from scipy.spatial import KDTree as KDTree_sp
from time import sleep, time


def potential_matches(hals1_pos, hals2_pos, box_size=1000):
    t_now = time()
    kdt = KDTree_skl(hals2_pos)
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'kdtree constructed')
    idx21 = kdt.query(hals1_pos, k=20, return_distance=False, dualtree=True, breadth_first=False)
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'query done for spatial neighbours')
    return idx21

def isin(x,s):
    return x in s

isin_v = np.vectorize(isin, excluded={1})

def in1d_searchsorted(A,B,assume_unique=False):
    if assume_unique==0:
        B_ar = np.unique(B)
    else:
        B_ar = B
    idx = np.searchsorted(B_ar,A)
    idx[idx==len(B_ar)] = 0
    return B_ar[idx] == A

def isinint(x_ar,y_ar):
    x,y = np.sort(x_ar), np.sort(y_ar)
    mina = min(x.min(), y.min())
    maxa = int(max(x.max(), y.max()) + 1)
    x = x - mina
    y = y - mina
    range_comb = int(maxa-mina)
    print(maxa, type(maxa), type(x))
    bool_arx = np.zeros(range_comb, dtype='bool')
    print(mina, maxa, bool_arx.size)
    bool_arx[x] = True
    bool_ary = np.zeros(range_comb, dtype='bool')
    bool_ary[y] = True
    return bool_arx & bool_ary #np.count_nonzero()
    

def matching_frac(pid1, pid2, max_num = 1000):
    t_now = time()
    pid1_smpl = np.random.choice(pid1, max_num, replace=False)
    pid2_smpl = np.random.choice(pid2, max_num, replace=False)
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'sample taken')
    pid1_smpl.sort(); pid2_smpl.sort(); pid1.sort(); pid2.sort()
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'sorted')
    # print(pid1_smpl)
    # match21_1 = np.count_nonzero(np.isin(pid1_smpl[:max_num//2], pid2, assume_unique=True)) 
    # match21_2 = np.count_nonzero(np.isin(pid1_smpl[max_num//2:], pid2, assume_unique=True))
    match21 = np.count_nonzero(np.isin(pid1_smpl, pid2, assume_unique=True))
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'np isin count_nonzero done')
    # match21 = np.count_nonzero(isinint(pid1_smpl, pid2))
    # t_bef, t_now = t_now, time()
    # print(t_now-t_bef, 'my isin count_nonzero done')
    # match21_py = len(set(pid1_smpl)&set(pid2))
    # t_bef, t_now = t_now, time()
    # print(t_now-t_bef, 'py set intersect')
    # print(match21, match21_py)
    match12 = np.count_nonzero(np.in1d(pid2_smpl, pid1, assume_unique=True))
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'np in1d count_nonzero done', match12)
    match12 = np.count_nonzero(in1d_searchsorted(pid2_smpl, pid1, assume_unique=True))
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'searchsort in1d count_nonzero done', match12)
    frac1 = match21 / pid1_smpl.size
    frac2 = match12 / pid2_smpl.size
    return (frac1,frac2)


# def matching_frac1(pid1, pid2, max_num = 1000):
#     pid1_smpl = np.random.choice(pid1, max_num, replace=False)
#     pid2_smpl = np.random.choice(pid2, max_num, replace=False)
#     pid1_smpl.sort(); pid1_smpl.sort(); pid1.sort(); pid2.sort()
#     print(pid1_smpl)
#     match21_1 = len(set(pid1_smpl[:max_num//2])&set(pid2)) 
#     match21_2 = len(set(pid1_smpl[max_num//2:])&set(pid2))
#     match21 = np.count_nonzero(np.isin(pid1_smpl, pid2, assume_unique=True))
#     print(match21, match21_1+match21_2)
#     frac1 = match21 / pid1_smpl.size
#     frac2 = np.count_nonzero(np.in1d(pid2_smpl, pid1, assume_unique=True))/ pid2_smpl.size
#     return (frac1,frac2)


def findin_rs(hal_vr_this, hal_rs_near_in):
    hal_rs_near_this = hal_rs_near_in[['X','Y','Z','VX','VY','VZ','Rvir']].copy()
    hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.X-hal_vr_this.Xc)<hal_vr_this.R_BN98]
    hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.Y-hal_vr_this.Yc)<hal_vr_this.R_BN98]
    hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.Z-hal_vr_this.Zc)<hal_vr_this.R_BN98]
#     if match_vel==True:
    hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VX-hal_vr_this.VXc)<np.abs(hal_vr_this.VXc)]
    hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VY-hal_vr_this.VYc)<np.abs(hal_vr_this.VYc)]
    hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VZ-hal_vr_this.VZc)<np.abs(hal_vr_this.VZc)]
    if hal_rs_near_this.shape[0]!=0:
        hal_rs_near_in['rel_size'] = np.abs(np.log(hal_rs_near_this.Rvir/1e3/hal_vr_this.R_BN98))
        return hal_rs_near_this.rel_size.idxmin()
    

    
def cross_match_old(hal_vr, hal_rs, dist_fac=0.5, match_vel=True):
    t_now = time()
    kdt = KDTree_skl(hal_rs[['X','Y','Z']].to_numpy())
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'kdtree constructed')
    neighbr = kdt.query_radius(hal_vr[['Xc','Yc','Zc']].to_numpy(), dist_fac*hal_vr.R_BN98.to_numpy())
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'query done for spatial neighbours')
    matched_vr_int_idx = []
    matched_rs_idx = []
    for i in range(hal_vr.shape[0]):
        hal_vr_this_Rvir = hal_vr.R_BN98.iloc[i]
        t_bef, t_now = t_now, time()
#         print(t_now-t_bef, 'for this vr halo')
        hal_rs_near_this = hal_rs[['VX','VY','VZ','Rvir']].iloc[neighbr[i]].copy()
        t_bef, t_now = t_now, time()
#         print(t_now-t_bef, 'for this vr halo, neighbor halos from index')
        if match_vel==True:
            hal_vr_this = hal_vr[['VXc','VYc','VZc','R_BN98']].iloc[i]
            hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VX-hal_vr_this.VXc)<np.abs(hal_vr_this.VXc)]
            hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VY-hal_vr_this.VYc)<np.abs(hal_vr_this.VYc)]
            hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VZ-hal_vr_this.VZc)<np.abs(hal_vr_this.VZc)]
        if hal_rs_near_this.shape[0]!=0:
            hal_rs_near_this_Rvir = hal_rs_near_this.Rvir.to_numpy()
            hal_rs_near_this_idxs = hal_rs_near_this.index.to_numpy()
            hal_rs_near_this_idx = np.argmin(np.abs(np.log(hal_rs_near_this_Rvir/1e3/hal_vr_this_Rvir)))
            matched_rs_idx.append(hal_rs_near_this.index[hal_rs_near_this_idx])
            matched_vr_int_idx.append(i)
            t_bef, t_now = t_now, time()
#             print(t_now-t_bef, 'best match selected')
    matched_vr_idx = hal_vr.index[matched_vr_int_idx]
#     matched_rs_idx = hal_vr.index[matched_rs_int_idx]
    t_bef, t_now = t_now, time()
    print(t_now-t_bef)
    return pd.DataFrame(data={'vr':matched_vr_idx,'rs':matched_rs_idx})

def cross_match_metric_vr(hal_vr, hal_rs, box_size, dist_fac=2, vel_offset=3000, metric_vel=0.01, metric_lograd=1):
    t_now = time()
    hal_rs_phase_sp = hal_rs[['X','Y','Z','VX','VY','VZ','Rvir']].to_numpy()
#     print(hal_rs_phase_sp)
    hal_rs_phase_sp[:,3:6] += vel_offset #hal_rs_phase_sp[:,3:6].min()
    hal_rs_phase_sp[:,3:6] *= metric_vel
    hal_rs_phase_sp[:,6] = np.log10(hal_rs_phase_sp[:,6])+1
    hal_rs_phase_sp[:,6] *= metric_lograd
#     print(hal_rs_phase_sp)
    kdt = KDTree_sp(hal_rs_phase_sp, boxsize=(box_size,)*6+(1e6,))
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'kdtree constructed')
    hal_vr_phase_sp = hal_vr[['Xc','Yc','Zc','VXc','VYc','VZc','R_BN98']].to_numpy(copy=True)
    hal_vr_phase_sp[:,3:6] += vel_offset # hal_vr_phase_sp[:,3:6].min()
    hal_vr_phase_sp[:,3:6] *= metric_vel
    hal_vr_phase_sp[:,6] = np.log10(hal_vr_phase_sp[:,6])+4
    hal_vr_phase_sp[:,6] *= metric_lograd
    neighbr_dist, neighbr = kdt.query(hal_vr_phase_sp, 1, workers=16) # distance_upper_bound=dist_fac*hal_vr.R_BN98.mean(), 
#     print(hal_vr_phase_sp, neighbr_dist)
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'query done for spatial neighbours')
    matched_rs_int_idx = neighbr #.T[0]
    match_found_this_vr = np.where(neighbr_dist<dist_fac*hal_vr.R_BN98.to_numpy(copy=True))
    matched_vr_idx = hal_vr.index[match_found_this_vr]
    matched_rs_idx = hal_rs.index[matched_rs_int_idx[match_found_this_vr]]
    t_bef, t_now = t_now, time()
    print(t_now-t_bef)
    return pd.DataFrame(data={'vr':matched_vr_idx.to_numpy(),'rs':matched_rs_idx.to_numpy()})


def cross_match_vr(hal_vr, hal_rs, dist_fac=0.5, match_vel=True):
    t_now = time()
    kdt = KDTree_sp(hal_rs[['X','Y','Z']].to_numpy())
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'kdtree constructed')
    neighbr = kdt.query(hal_vr[['Xc','Yc','Zc']].to_numpy(), 10, distance_upper_bound=dist_fac*hal_vr.R_BN98.mean())[1]
    t_bef, t_now = t_now, time()
    print(t_now-t_bef, 'query done for spatial neighbours')
    matched_vr_int_idx = []
    matched_rs_idx = []
    for i in range(hal_vr.shape[0]):
        hal_vr_this_Rvir = hal_vr.R_BN98.iloc[i]
        t_bef, t_now = t_now, time()
        print(t_now-t_bef, 'for this vr halo')
        neighbr = neighbr[neighbr<hal_vr.shape[0]]
        hal_rs_near_this = hal_rs[['VX','VY','VZ','Rvir']].iloc[neighbr[i]].copy()
        t_bef, t_now = t_now, time()
        print(t_now-t_bef, 'for this vr halo, neighbor halos from index')
        if match_vel==True:
            hal_vr_this = hal_vr[['VXc','VYc','VZc','R_BN98']].iloc[i]
            hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VX-hal_vr_this.VXc)<np.abs(hal_vr_this.VXc)]
            hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VY-hal_vr_this.VYc)<np.abs(hal_vr_this.VYc)]
            hal_rs_near_this = hal_rs_near_this.loc[np.abs(hal_rs_near_this.VZ-hal_vr_this.VZc)<np.abs(hal_vr_this.VZc)]
        if hal_rs_near_this.shape[0]!=0:
            hal_rs_near_this_Rvir = hal_rs_near_this.Rvir.to_numpy()
            hal_rs_near_this_idxs = hal_rs_near_this.index.to_numpy()
            hal_rs_near_this_idx = np.argmin(np.abs(np.log(hal_rs_near_this_Rvir/1e3/hal_vr_this_Rvir)))
            matched_rs_idx.append(hal_rs_near_this.index[hal_rs_near_this_idx])
            matched_vr_int_idx.append(i)
            t_bef, t_now = t_now, time()
            print(t_now-t_bef, 'best match selected')
    matched_vr_idx = hal_vr.index[matched_vr_int_idx]
#     matched_rs_idx = hal_vr.index[matched_rs_int_idx]
    t_bef, t_now = t_now, time()
    print(t_now-t_bef)
    return pd.DataFrame(data={'vr':matched_vr_idx,'rs':matched_rs_idx})

# def select_best_size():
    








