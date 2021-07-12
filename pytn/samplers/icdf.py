#%%
import numpy as np
import scipy.stats as st
from scipy.special import erf
from scipy.interpolate import interp1d

# %%
from numpy.random import default_rng


# %%
import matplotlib.pyplot as plt
#%%
def sample_cdf(N, cdf,  dens_interval=(-5,5), use_scipy=False, seed=None):
    if use_scipy==True:
        rvc = st.rv_continuous()
        rvc._cdf = cdf
        return rvc.rvs(size=N)
    else:
        r = np.linspace(*dens_interval, 100000)
        icdf = interp1d(cdf(r),r)
        rng = default_rng(seed=seed)
        return icdf(rng.random(N))



# %%
if __name__=="__main__":
    smpl1 = sample_cdf(N=1000, cdf=lambda x : erf(x)/2+0.5)

    smpl2 = sample_cdf(N=100000, cdf=lambda x : erf(x)/2+0.5, use_scipy=False)

    plt.hist(smpl1, bins=100, density=True, histtype='step')[-1]
    plt.hist(smpl2, bins=1000, density=True, histtype='step')[-1]
    # %timeit rng.random(10000)
    # %timeit np.random.random(10000)

# %%
