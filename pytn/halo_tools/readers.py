import numpy as np
import os, re
import pandas as pd

#%%
def read_rockstar(filepath):
    with open(filepath) as f:
        header_line = f.readline()
    cols = ['('.join(col.split('(')[:-1]) for col in re.split('\s+',header_line[1:])]
    # hals = pd.read_csv(filepath, sep=r'\s+', comment='#', engine='c', names=cols)
    # hals.set_index('id')
    return pd.read_csv(filepath, sep=r'\s+', comment='#', engine='c', names=cols).set_index('id')