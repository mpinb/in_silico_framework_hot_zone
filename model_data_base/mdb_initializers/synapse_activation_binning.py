'''
Methods for fast binning of synapse activation dask dataframe.

Exemplary use:

    groupby  = ['EI']
    values = synapse_activation_postprocess_dask(mdb['synapse_activation'], groupby = groupby, prefun = prefun, applyfun = applyfun, postfun = postfun)
    values = values.compute(get = c.get)
    print 'save'
    save_groupby(mdb, values, groupby)
'''

from __future__ import absolute_import
from collections import defaultdict
import numpy as np
import dask
from ..analyze import excitatory, inhibitory
from ..analyze.temporal_binning import universal as temporal_binning
from ..IO.LoaderDumper import numpy_to_msgpack as numpy_to_msgpack

maxtime = 400

def prefun(df):
    dummy = df.synapse_type.str.split('_')
    df['celltype'] = dummy.str[0]
    df['presynaptic_column'] = dummy.str[1]
    df['proximal'] = (df.soma_distance < 500).replace(True, 'prox').replace(False, 'dist')
    df['EI'] = df['celltype'].isin(excitatory).replace(True, 'EXC').replace(False, 'INH')
    bs = 50
    df['binned_somadist'] = df.soma_distance.div(bs).map(np.floor).astype(int).map(lambda x: '{}to{}'.format(x*bs, x*bs+bs))    
    return df

def postfun(s):
    defaultvalue = np.zeros(maxtime)
    s = s.map(lambda x: defaultvalue if x is None else x)
    return np.vstack(s.values)

def applyfun(pdf):
    return temporal_binning(pdf, min_time = 0, max_time = maxtime, normalize = False)[1]



def synapse_activation_postprocess_pandas(pdf, groupby = '', \
                                          prefun = None, 
                                          applyfun = None, 
                                          postfun = None):
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

def tree_reduction(delayeds, aggregate_fun, length = 7):
    if len(delayeds) > length:
        chunks = [delayeds[i:i+length] for i  in range(0, len(delayeds), length)]
        delayeds = [aggregate_fun(chunk) for chunk in chunks]
        return tree_reduction(delayeds, aggregate_fun, length)
    else:
        return aggregate_fun(delayeds)
    
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
                'binned_somadist' synapse counts for all 50 microns
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
    ret = tree_reduction(ds, merge_results_together)
    
    if mdb is not None:
        assert('groupby' in kwargs)
        data = ret.compute(get = get)
        save_groupby(mdb, data, kwargs['groupby'])
    else:
        return ret

def save_groupby(mdb, result, groupby):
    '''saves the result of synapse_activation_postprocess_dask to a model data base.
    
    A new model data base within mdb is created and the numpy arrays are stored there.'''
    if not isinstance(groupby, list): groupby = [groupby]    
    identifier = tuple(['synapse_activation_binned', 't1'] + ['__'.join(groupby)])
    try:
        del mdb[identifier]
    except:
        pass
    sub_mdb = mdb.create_sub_mdb(identifier)
    for key in result:
        sub_mdb.setitem(key, result[key], dumper = numpy_to_msgpack)
        
def init(mdb, groupby = '', get = None, 
         prefun = prefun, 
         applyfun = applyfun, 
         postfun = postfun):
    '''
    Binning synapse activations.
    
    mdb: ModelDataBase object, which is already initalized, such that 
         the key mdb['synapse_activation'] exists.
    
    groupby: (default: ''): species for which subgroups the bins should be 
            calculated. Available values include: 
                'celltype', 
                'presynaptic_column', 
                'proximal', (soma distance < 500 ym)
                'EI' (Lumping the EXC / INH celltypes together)
            It can also be any column in the specified dataframe.
            Can be a list, if "sub-subgroups" should be calculated.

    get: allows to specify a dask scheduler for the computation
    
    Not implemented yet:
        (prefun: function to apply on each partition before binning)
        (applyfun: actual binning function)
        (postfun: function to merge results together for one partition)
        
    
    returns: None. The binned synapse activation data will be stored in mdb.
    '''
    
    synapse_activation_postprocess_dask(mdb['synapse_activation'], \
                                        groupby = groupby, mdb = mdb, \
                                        get = get, \
                                        prefun = prefun, \
                                        applyfun = applyfun, \
                                        postfun = postfun)
