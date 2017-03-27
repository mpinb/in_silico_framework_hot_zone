from .context import *
from model_data_base.model_data_base import ModelDataBase
from . import decorators
import unittest
import os, shutil
import numpy as np

class Tests(unittest.TestCase):       
    def setUp(self):        
        self.mdb = ModelDataBase(test_mdb_folder)

    ### todo: most of the tests is actually testing the mdb initialization
    @decorators.testlevel(0)    
    def test_no_empy_rows(self):
        e = self.mdb
        synapse_activation = e['synapse_activation']
        synapse_activation['isnan'] = synapse_activation.soma_distance.apply(lambda x: np.isnan(x))
        nr_nan_rows_synapse_activation = len(synapse_activation[synapse_activation.isnan == True])
        self.assertEqual(0, nr_nan_rows_synapse_activation)
        
    @decorators.testlevel(1)            
    def test_dataintegrity_no_empty_rows(self):
        e = self.mdb
        synapse_activation = e['synapse_activation']
        cell_activation = e['cell_activation']
        voltage_traces = e['voltage_traces']
        synapse_activation['isnan']=synapse_activation.soma_distance.apply(lambda x: np.isnan(x))
        cell_activation['isnan']=cell_activation['0'].apply(lambda x: np.isnan(x)) 
        first_column = e['voltage_traces'].columns[0]
        voltage_traces['isnan']=voltage_traces[first_column].apply(lambda x: np.isnan(x)) 
         
        self.assertEqual(0, len(synapse_activation[synapse_activation.isnan == True]))
        self.assertEqual(0, len(cell_activation[cell_activation.isnan == True]))
        self.assertEqual(0, len(voltage_traces[voltage_traces.isnan == True]))
 
    def test_voltage_traces_have_float_indices(self):
        e = self.mdb
        self.assertIsInstance(e['voltage_traces'].columns[0], float)
        self.assertIsInstance(e['voltage_traces'].head().columns[0], float)

## todo        
#     def test_maybe_calculate(self):
#         #allways calls function, if force_calculate == True
#         #calls function only if unknown, if force_calculate = False
#         #passes arguments to dumper
#         pass
         
         