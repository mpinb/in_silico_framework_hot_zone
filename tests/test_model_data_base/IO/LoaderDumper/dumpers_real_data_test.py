from model_data_base.model_data_base import ModelDataBase
from ... import decorators
import tempfile
import numpy as np
from pandas.util.testing import assert_frame_equal
import dask.dataframe as dd
import pandas as pd
import dask
from  model_data_base.IO.LoaderDumper import dask_to_csv, numpy_to_npy, pandas_to_msgpack, \
                            to_pickle, pandas_to_pickle, dask_to_msgpack, \
                            dask_to_categorized_msgpack, to_cloudpickle, reduced_lda_model
from test_simrun2.reduced_model.get_kernel_test import get_test_Rm
from numpy.testing import assert_array_equal
import distributed
import tempfile

def robust_del_fun(mdb, key):
    try:
        del mdb[key]
    except KeyError:
        pass
        

def real_data_generic(fresh_mdb, dumper, client = None):
    if client is None:
        fresh_mdb.setitem('voltage_traces2', fresh_mdb['voltage_traces'], dumper = dumper)
    else:
        fresh_mdb.setitem('voltage_traces2', fresh_mdb['voltage_traces'], dumper = dumper, client = client)
    dummy = fresh_mdb['voltage_traces2']
    b = fresh_mdb['voltage_traces'].compute(get = dask.multiprocessing.get)
    a = dummy.compute(get = dask.multiprocessing.get)
    assert_frame_equal(a, b)   
    
    if client is None:
        fresh_mdb.setitem('synapse_activation2', fresh_mdb['synapse_activation'], dumper = dumper)
    else:
        fresh_mdb.setitem('synapse_activation2', fresh_mdb['synapse_activation'], dumper = dumper, client = client)
    dummy = fresh_mdb['synapse_activation2']
    b = fresh_mdb['synapse_activation'].compute(get = dask.multiprocessing.get)
    a = dummy.compute(get = dask.multiprocessing.get)
    assert_frame_equal(a, b)        

def test_dask_to_csv_real_data(client, fresh_mdb):
    real_data_generic(fresh_mdb, dask_to_csv, client=client)

def test_dask_to_categorized_msgpack_real_data(client, fresh_mdb):
    real_data_generic(fresh_mdb, dask_to_categorized_msgpack, client = client)        

def test_dask_to_msgpack_real_data(client, fresh_mdb):
    real_data_generic(fresh_mdb, dask_to_msgpack, client = client)
    