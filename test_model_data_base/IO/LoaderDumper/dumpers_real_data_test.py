from ...context import *
from model_data_base.model_data_base import ModelDataBase
from ... import decorators
import unittest
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


def robust_del_fun(mdb, key):
    try:
        del mdb[key]
    except KeyError:
        pass
            
assert client is not None 

class Tests(unittest.TestCase): 
    def setUp(self):
        # set up model_data_base in temporary folder and initialize it.
        # This additionally is an implicit test, which ensures that the
        # initialization routine does not throw an error.
        self.fmdb = FreshlyInitializedMdb()
        self.mdb = self.fmdb.__enter__()
                       
    def tearDown(self):
        self.fmdb.__exit__()
                    
    def real_data_generic(self, dumper, client = None):
        if client is None:
            self.mdb.setitem('voltage_traces2', self.mdb['voltage_traces'], dumper = dumper)
        else:
            self.mdb.setitem('voltage_traces2', self.mdb['voltage_traces'], dumper = dumper, client = client)
        dummy = self.mdb['voltage_traces2']
        b = self.mdb['voltage_traces'].compute(get = dask.multiprocessing.get)
        a = dummy.compute(get = dask.multiprocessing.get)
        assert_frame_equal(a, b)   
        
        if client is None:
            self.mdb.setitem('synapse_activation2', self.mdb['synapse_activation'], dumper = dumper)
        else:
            self.mdb.setitem('synapse_activation2', self.mdb['synapse_activation'], dumper = dumper, client = client)
        dummy = self.mdb['synapse_activation2']
        b = self.mdb['synapse_activation'].compute(get = dask.multiprocessing.get)
        a = dummy.compute(get = dask.multiprocessing.get)
        assert_frame_equal(a, b)        

    @decorators.testlevel(2)
    def test_dask_to_csv_real_data(self):
        self.real_data_generic(dask_to_csv)

    @decorators.testlevel(2)
    def test_dask_to_categorized_msgpack_real_data(self):
        self.real_data_generic(dask_to_categorized_msgpack, client = client)        

    @decorators.testlevel(2)
    def test_dask_to_msgpack_real_data(self):
        self.real_data_generic(dask_to_msgpack, client = client)
        