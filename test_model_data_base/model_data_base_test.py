from .context import *
from model_data_base.model_data_base import ModelDataBase
from . import decorators
import unittest
import os, shutil
import numpy as np

class Tests(unittest.TestCase):       
    def setUp(self):        
        self.prefix = os.path.dirname(os.path.abspath(__file__))     
        self.empty_folder = os.path.join(self.prefix, "data/empty_folder")
        self.test_temp = os.path.join(self.prefix, "data/test_temp")
        self.test_data = os.path.join(self.prefix, 'data/test_data')      
        self.nonexistent = os.path.join(self.prefix, 'data/nonexistent') 

        mdb = ModelDataBase(self.test_temp)
        mdb.settings.show_computation_progress = False
        if not 'synapse_activation' in mdb.keys():
            import model_data_base.mdb_initializers.load_roberts_simulationdata
            print self.test_data
            model_data_base.mdb_initializers.load_roberts_simulationdata.init(mdb, self.test_data)     
        if not 'spike_times' in mdb.keys():
            import model_data_base.mdb_initializers.load_roberts_simulationdata            
            model_data_base.mdb_initializers.load_roberts_simulationdata.pipeline(mdb)    
         
        shutil.rmtree(self.empty_folder)
        os.mkdir(self.empty_folder)
        try:
            shutil.rmtree(self.nonexistent)
        except(OSError):
            pass
    
    @decorators.testlevel(1)    
    def test_canbeinstanciated(self):
        ModelDataBase(self.test_temp)

    @decorators.testlevel(0)    
    def test_no_empy_rows(self):
        e = ModelDataBase(os.path.join(self.test_temp))
        synapse_activation = e['synapse_activation']
        synapse_activation['isnan'] = synapse_activation.soma_distance.apply(lambda x: np.isnan(x))
        nr_nan_rows_synapse_activation = len(synapse_activation[synapse_activation.isnan == True])
        self.assertEqual(0, nr_nan_rows_synapse_activation)
        
    @decorators.testlevel(1)            
    def test_dataintegrity_no_empty_rows(self):
        e = ModelDataBase(self.test_temp)
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
        e = ModelDataBase(self.test_temp)
        self.assertIsInstance(e['voltage_traces'].columns[0], float)
        self.assertIsInstance(e['voltage_traces'].head().columns[0], float)
        
    def test_sqlitedict(self):
        e = ModelDataBase(self.test_temp)
        x = np.random.rand(10)
        e['sqltest']=x
        y = e['sqltest']
        np.testing.assert_array_equal(x, y, "there was an error reading from the sqlitedict")
        
#     def test_maybe_calculate(self):
#         #allways calls function, if force_calculate == True
#         #calls function only if unknown, if force_calculate = False
#         #passes arguments to dumper
#         pass
         
         