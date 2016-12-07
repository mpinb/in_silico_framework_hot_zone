'''
Created on Aug 15, 2016

@author: arco
'''
import os, glob, sys
import numpy as np
import pandas as pd
import dask.dataframe, dask.delayed
import unittest
def read_voltage_traces_from_file(prefix, fname):
    full_fname = os.path.join(prefix, fname)
    data = np.loadtxt(full_fname, skiprows=1, unpack=True, dtype = 'float64')
    t = data[0]
    data = data[1:]
    index=[str(os.path.join(fname, str(index))) for index in range(len(data))]
    df = pd.DataFrame(data, columns=t)
    df['sim_trail_index'] = index
    df.set_index('sim_trail_index', inplace = True)
    return df

def read_voltage_traces(prefix, fnames):
    '''takes list of filenames, returns dask dataframe'''
    out = []
    for fname in fnames:
        out.append(dask.delayed(read_voltage_traces_from_file)(prefix, fname))
    out = dask.dataframe.from_delayed(out, meta=out[0].compute())    
    return out