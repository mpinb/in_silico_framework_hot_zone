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
from model_data_base.analyze import excitatory, inhibitory
#general function
from collections import defaultdict

def prefun(df):
    dummy = df.synapse_type.str.split('_')
    df['celltype'] = dummy.str[0]
    df['presynaptic_column'] = dummy.str[1]
    df['proximal'] = (df.soma_distance < 500).replace(True, 'prox').replace(False, 'dist')
    df['EI'] = df['celltype'].isin(excitatory).replace(True, 'EXC').replace(False, 'INH')
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
    '''see docstring of synapse_activation_postprocess_dask'''

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
    '''
    Calculates bins of synapse activation dask dataframe per trail.
    #Todo: make this method out of core
    
    args:
        ddf: synapse activation dask dataframe
    kwargs:
        groupby: (default: ''): species for which subgroups the bins should be 
            calculated. Available values include: 
                'celltype', 
                'presynaptic_column', 
                'proximal', (soma distance < 500 ym)
                'EI' (Lumping the EXC / INH celltypes together)
            It can also be any column in the specified dataframe.
            Can be a list, if "sub-subgroups" should be calculated.
        mdb: if specified, the result will be computed immediately and 
                saved in the database immediately.
        get: only has an effect if 'mdb' kwarg is provided. In this case,
                it allows to specify a dask scheduler for the computation
        (prefun: function to apply on each partition before binning)
        (applyfun: actual binning function)
        (postfun: function to merge results together for one partition)
        
    
    returns: 
        dask.delayed object. If computed, this will return a dictionary 
        containing numpy arrays. rows: sim trails, columns: bins
    '''
    fun = dask.delayed(synapse_activation_postprocess_pandas)
    ds = ddf.to_delayed()
    
    #special case: if mdb is defined: isolate that keyword 
    #for later use
    if 'mdb' in kwargs:
        mdb = kwargs['mdb']
        del kwargs['mdb']
    else:
        mdb = None
    if 'get' in kwargs:
        get = kwargs['get']
        del kwargs['get']
    else:
        get = None        
         
    ds = [fun(d, **kwargs) for d in ds]
    ret = dask.delayed(merge_results_together(ds))
    
    if mdb is not None:
        assert('groupby' in kwargs)
        data = ret.compute(get = get)
        save_groupby(mdb, data, kwargs['groupby'])
    else:
        return ret

def save_groupby(mdb, result, groupby):
    '''saves the result of synapse_activation_postprocess_dask to a model data base.
    
    A new model data base within mdb is created and the numpy arrays are stored there.'''
    identifier = tuple(['synapse_activation', 'binned_t1'] + groupby)
    try:
        del mdb[identifier]
    except:
        pass
    sub_mdb = mdb.get_sub_mdb(identifier)
    for key in result:
        sub_mdb.setitem(key, result[key], dumper = I.dumper_numpy_to_npy)
        
