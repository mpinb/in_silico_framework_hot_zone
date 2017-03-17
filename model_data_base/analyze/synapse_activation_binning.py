'''
Methods for fast binning of synapse activation dask dataframe.

Exemplary use:

    groupby  = ['EI']
    values = synapse_activation_postprocess_dask(mdb['synapse_activation'], groupby = groupby, prefun = prefun, applyfun = applyfun, postfun = postfun)
    values = values.compute(get = c.get)
    print 'save'
    save_groupby(mdb, values, groupby)
'''


import numpy as np
import dask
import Interface as I

#general function
from collections import defaultdict

def prefun(df):
    dummy = df.synapse_type.str.split('_')
    df['celltype'] = dummy.str[0]
    df['presynaptic_column'] = dummy.str[1]
    df['proximal'] = (df.soma_distance < 500).replace(True, 'prox').replace(False, 'dist')
    df['EI'] = df['celltype'].isin(I.excitatory).replace(True, 'EXC').replace(False, 'INH')
    return df

def postfun(s):
    defaultvalue = np.zeros(300)
    s = s.map(lambda x: defaultvalue if x is None else x)
    return np.vstack(s.values)

def applyfun(pdf):
    return I.temporal_binning(pdf, min_time = 0, max_time = 300, normalize = False)[1]



def synapse_activation_postprocess_pandas(pdf, groupby = '', \
                                          prefun = prefun, 
                                          applyfun = applyfun, 
                                          postfun = postfun):

    if not isinstance(groupby, list): groupby = [groupby]
    pdf = prefun(pdf)
    groups = pdf.groupby([pdf.index] + groupby).apply(applyfun)
    
    for lv in range(len(groupby)):
        groups = groups.unstack(1)
    keys = list(groups.columns)
    out = {key: postfun(groups[key]) for key in keys}
    return out

@dask.delayed
def merge_results_together(dicts):
    out = defaultdict(lambda: [])
    all_keys = set([x for d in dicts for x in d.keys()])
    for d in dicts:
        for key in all_keys:#d.keys():
            if key in d:
                out[key].append(d[key])
            else:
                out[key].append(np.zeros(d[d.keys()[0]].shape)) #fill with zeros

    for key in out.keys():
        out[key] = np.vstack(out[key])
    return out
    
def synapse_activation_postprocess_dask(ddf, **kwargs):
    fun = dask.delayed(synapse_activation_postprocess_pandas)
    ds = ddf.to_delayed()
    ds = [fun(d, **kwargs) for d in ds]
    return dask.delayed(merge_results_together(ds))




def save_groupby(mdb, result, groupby):
    identifier = tuple(['synapse_activation', 'binned_t1'] + groupby)
    try:
        del mdb[identifier]
    except:
        pass
    sub_mdb = mdb.get_sub_mdb(identifier)
    for key in result:
        sub_mdb.setitem(key, result[key], dumper = I.dumper_numpy_to_npy)
        
