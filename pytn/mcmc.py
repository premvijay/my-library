# %%

import numpy as np
from numpy import random

import matplotlib.pyplot as plt

def normal(x,mu,sigma):
    numerator = np.exp((-(x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)
    return numerator/denominator

def random_coin(p):
    unif = random.uniform(0,1)
    if unif>=p:
        return False
    else:
        return True
    
def gaussian_mcmc(hops,mu,sigma):
    states = []
    burn_in = int(hops*0.2)
    current = random.uniform(-5*sigma+mu,5*sigma+mu)
    for i in range(hops):
        states.append(current)
        walk = random.uniform(-5*sigma+mu,5*sigma+mu)
        
        curr_prob = normal(x=current,mu=mu,sigma=sigma)
        move_prob = normal(x=walk,mu=mu,sigma=sigma)
        
        acceptance = min(move_prob/curr_prob,1)
        if random_coin(acceptance):
            current = walk
    return states[burn_in:]

# %%
def sample_mpolis(hops,pdf, scale=5):
    states = []
    burn_in = int(hops*0.2)
    mark_kernel = lambda : random.uniform(-scale,scale)
    # mark_kernel = lambda : random.standard_normal(1)
    current = mark_kernel()
    for i in range(hops):
        states.append(current)
        walk = current + mark_kernel()
        
        curr_prob = pdf(current)
        walk_prob = pdf(walk)
        
        acceptance = min(walk_prob/curr_prob,1)
        if random_coin(acceptance):
            current = walk
    return np.asarray(states[burn_in:])

def gaussian_mcmc(hops,mu,sigma):
    pdf = lambda x : normal(x,mu=mu,sigma=sigma)
    return sample_mpolis(hops,pdf)
    

# %%
if __name__=="__main__":
    pdf = lambda x : normal(x,mu=0,sigma=1)
    dist = sample_mpolis(100000,pdf)
    lines = np.linspace(-3,3,1000)
    plt.hist(dist, density=True,bins=50) 
    plt.plot(lines, pdf(lines))

#%%
if __name__=="__main__":
    def pdf(x):
        return np.where( abs(x)<10, abs(x)/100,0)
# %%
if __name__=="__main__":
    dist = sample_mpolis(100000,pdf)
    lines = np.linspace(-15,15,1000)
    plt.hist(dist, density=True,bins=50) 
    plt.plot(lines, pdf(lines))
# %%
