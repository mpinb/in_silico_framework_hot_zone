from ...context import *
from model_data_base.model_data_base import ModelDataBase
from ... import decorators
import unittest
import numpy as np
from pandas.util.testing import assert_frame_equal
import dask.dataframe as dd
import pandas as pd
import dask
from  model_data_base.IO.LoaderDumper import dask_to_csv, numpy_to_npy, pandas_to_msgpack, \
                                to_pickle, pandas_to_pickle, dask_to_msgpack, \
                                dask_to_categorized_msgpack, to_cloudpickle
from numpy.testing import assert_array_equal


def robust_del_fun(mdb, key):
    try:
        del mdb[key]
    except KeyError:
        pass
            
class Tests(unittest.TestCase):       
    def setUp(self): 
        mdb = ModelDataBase(test_mdb_folder) 
        self.mdb = mdb 
        
        self.pdf = pd.DataFrame({0: [1,2,3,4,5,6], 1: ['1', '2', '3', '1', '2', '3'], '2': [1, '2', 3, 1, '2', 3], \
                                 'myname': ['bla', 'bla', 'bla', 'bla', 'bla', 'bla']})  
        self.ddf = dd.from_pandas(self.pdf, npartitions = 2)
        #for to_csv methods, since it cannot provide support for mixed dtypes
        self.pdf2 = pd.DataFrame({0: [1,2,3,4,5,6], 1: ['1', '2', '3', '1', '2', '3'], '2': [1, 2, 3, 1, 2, 3], \
                                 'myname': ['bla', 'bla', 'bla', 'bla', 'bla', 'bla']})  
        self.ddf2 = dd.from_pandas(self.pdf2, npartitions = 2)
                    
        assert(self.ddf.npartitions > 1)
        
    def clean_up(self):
        robust_del_fun(self.mdb, 'synapse_activation2')
        robust_del_fun(self.mdb, 'voltage_traces2')
        robust_del_fun(self.mdb, 'test')       
        
    def data_frame_generic_small(self, pdf, ddf, dumper): 
        #index not set        
        self.clean_up()
        self.mdb.setitem('test', ddf, dumper = dumper)
        dummy = self.mdb['test']
        a = dask.compute(dummy)[0].reset_index(drop = True)
        b = pdf.reset_index(drop = True)
        assert_frame_equal(a, b)
        #sorted index set
        self.clean_up()
        self.mdb.setitem('test', ddf.set_index(0), dumper = dumper)
        dummy = self.mdb['test']
        a = dask.compute(dummy)[0]
        b = pdf.set_index(0)
        assert_frame_equal(a, b)  
        
    def test_dask_to_csv_small(self):
        self.data_frame_generic_small(self.pdf2, self.ddf2, dask_to_csv)
        
    def test_dask_to_msgpack_small(self):
        self.data_frame_generic_small(self.pdf2, self.ddf2, dask_to_msgpack)     
        
    def test_dask_to_categorized_msgpack_small(self):
        self.data_frame_generic_small(self.pdf2, self.ddf2, dask_to_categorized_msgpack)
        
    def test_pandas_to_msgpack_small(self):
        self.data_frame_generic_small(self.pdf, self.pdf, pandas_to_msgpack)
    
    def test_pandas_to_pickle_small(self):
        self.data_frame_generic_small(self.pdf, self.pdf, pandas_to_pickle)
        
    def test_to_pickle_small(self):
        self.data_frame_generic_small(self.pdf, self.pdf, to_pickle) 
        
    def test_to_cloudpickle_small(self):
        self.data_frame_generic_small(self.pdf, self.pdf, to_cloudpickle)         
               
    def test_self_small(self):
        self.data_frame_generic_small(self.pdf, self.pdf, 'self')         
        
    def test_numpy_to_npy(self):      
        def fun(x):
            self.clean_up()
            self.mdb.setitem('test', x, dumper = numpy_to_npy)
            dummy = self.mdb['test']
            assert_array_equal(dummy, x)
        fun(np.random.randint(5, size=(100, 100)))    
        fun(np.random.randint(5, size=(100,)))
        fun(np.array([]))
        
        
    def real_data_generic(self, dumper):
        self.mdb.setitem('voltage_traces2', self.mdb['voltage_traces'], dumper = dumper)
        dummy = self.mdb['voltage_traces2']
        b = self.mdb['voltage_traces'].compute(get = dask.multiprocessing.get)
        a = dummy.compute(get = dask.multiprocessing.get)
        assert_frame_equal(a, b)   
           
        self.mdb.setitem('synapse_activation2', self.mdb['synapse_activation'], dumper = dumper)
        dummy = self.mdb['synapse_activation2']
        b = self.mdb['synapse_activation'].compute(get = dask.multiprocessing.get)
        a = dummy.compute(get = dask.multiprocessing.get)
        assert_frame_equal(a, b)        

    @decorators.testlevel(2)
    def test_dask_to_csv_real_data(self):
        self.real_data_generic(dask_to_csv)

    @decorators.testlevel(2)
    def test_dask_to_categorized_msgpack_real_data(self):
        self.real_data_generic(dask_to_categorized_msgpack)        

    @decorators.testlevel(2)
    def test_dask_to_msgpack_real_data(self):
        self.real_data_generic(dask_to_msgpack)          
         

        
        