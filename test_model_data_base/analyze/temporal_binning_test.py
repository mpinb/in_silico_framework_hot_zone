from ..context import *
from .. import decorators
from model_data_base.analyze.temporal_binning import *
import unittest
import pandas as pd
import numpy as np
import dask.dataframe as dd
from model_data_base.settings import npartitions
from model_data_base.model_data_base import ModelDataBase

class Tests(unittest.TestCase):
    def setUp(self):
        self.pdf = pd.DataFrame({'blabla': ['a', 1, 3], \
                                '1': [1, 5, 10], \
                                '2': [15, 1, 30], \
                                '3': [np.NaN, 1, 45]})
    def test_temporal_binning_pandas(self):
        '''tests the temporal binning function with pandas'''
        bins, hist = temporal_binning_pd(self.pdf, bin_size = 10, min_time = 0, normalize = False)
        np.testing.assert_array_equal(bins, np.array([0,10,20,30,40,50]))
        np.testing.assert_array_equal(hist, np.array([4,2,0,1,1]))
        
        bins, hist = temporal_binning_pd(self.pdf, bin_size = 10, min_time = 0, normalize = True)
        np.testing.assert_array_equal(bins, np.array([0,10,20,30,40,50]))
        np.testing.assert_array_equal(hist, np.array([4/3.,2/3.,0/3.,1/3.,1/3.]))        
        
    def test_temporal_binning_dask(self):
        ddf = dd.from_pandas(self.pdf, npartitions = 3)
        bins, hist = temporal_binning_dask(ddf, bin_size = 10, min_time = 0, max_time = 50, normalize = False)
        np.testing.assert_array_equal(bins, np.array([0,10,20,30,40,50]))
        np.testing.assert_array_equal(hist, np.array([4,2,0,1,1])) 
    
    @decorators.testlevel(1)    
    def test_binning_real_data(self):
        with FreshlyInitializedMdb() as mdb:            
            pdf = mdb['spike_times']
        #if dask: convert to pandas
        try:
            pdf = pdf.compute()
        except:
            pass
        
        ddf = dd.from_pandas(pdf, npartitions = npartitions)
        t_bins_pandas, data_pandas = temporal_binning_pd(pdf, 1, 0, 300)
        t_bins_dask, data_dask = temporal_binning_dask(ddf, 1, 0, 300)
        
        #print data_dask
        np.testing.assert_equal(t_bins_pandas, t_bins_dask)
        np.testing.assert_equal(data_pandas, data_dask)
        
        
        
        