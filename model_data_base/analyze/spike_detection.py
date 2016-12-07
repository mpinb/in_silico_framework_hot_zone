import pandas as pd
import dask
import numpy as np

def simple_spike_detection(t, v, tBegin=None, tEnd=None, threshold=0.0, mode='regular'):
    '''
    copied from robert
    Simple spike detection method. Identify
    spike times within optional window [tBegin, tEnd]
    by determining threshold crossing times from below
    supported modes:
    regular: absolute threshold crossing
    differential: threshold crossing of dv/dt
    '''
    if len(t) != len(v):
        errstr = 'Dimensions of time vector and membrane potential vector not matching'
        raise RuntimeError(errstr)
    
    tSpike = []
    beginIndex = 1
    endIndex = len(t)
    if tBegin is not None:
        for i in range(1,len(t)):
            if t[i-1] < tBegin and t[i] >= tBegin:
                beginIndex = i
                break
    if tEnd is not None:
        for i in range(1,len(t)):
            if t[i-1] < tEnd and t[i] >= tEnd:
                endIndex = i
                break
    
    if mode == 'regular':
        for i in range(beginIndex,endIndex):
            if v[i-1] < threshold and v[i] >= threshold:
                tSpike.append(t[i])
    
    if mode == 'slope':
        dvdt = np.diff(v)/np.diff(t)
        for i in range(beginIndex,endIndex-1):
            if dvdt[i-1] < threshold and dvdt[i] >= threshold:
                tSpike.append(t[i])
    
    return tSpike


def _helper(x):
    '''reads out a voltage trace, so it can be fed into simple_spike_detection()
    and converts the result back to pd.Series, so the result can be concatenated 
    to a dask dataframe'''
    t = x.index.values
    values = x.values
    spikes = simple_spike_detection(t, values, mode = 'regular', threshold = 0)
    #print(len(spikes))
    return pd.Series({lv: x for lv, x in enumerate(spikes)})

def spike_detection(ddf):
    #print(type(mdb['voltage_traces']))
#     return mdb['voltage_traces'].apply(_helper, axis = 1)
    return ddf.apply(_helper, axis = 1)

