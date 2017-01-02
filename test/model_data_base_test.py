from .context import *
from model_data_base.model_data_base import ModelDataBase
from . import decorators
import unittest
import os, shutil
from mock import MagicMock
import numpy as np
#for x in sys.modules: print x

class Tests(unittest.TestCase):       
    def setUp(self):
        self.prefix = os.path.dirname(os.path.abspath(__file__))     
        self.empty_folder = os.path.join(self.prefix, "data/empty_folder")
        self.test_temp = os.path.join(self.prefix, "data/test_temp")
        self.test_data = os.path.join(self.prefix, 'data/test_data')      
        self.nonexistent = os.path.join(self.prefix, 'data/nonexistent') 

        mdb = ModelDataBase('test/data/test_temp') 
        if not 'synapse_activation' in mdb.keys():
            import model_data_base.mdb_initializers.load_roberts_simulationdata
            model_data_base.mdb_initializers.load_roberts_simulationdata.init(mdb, 'test/data/test_data')     
        if not 'spike_times' in mdb.keys():
            import model_data_base.mdb_initializers.load_roberts_simulationdata            
            model_data_base.mdb_initializers.load_roberts_simulationdata.pipeline(mdb)    
         
        shutil.rmtree(self.empty_folder)
        os.mkdir(self.empty_folder)
        try:
            shutil.rmtree(self.nonexistent)
        except(OSError):
            pass
         
#     def test_already_build(self):
#         '''check the already build function'''
#         e = ModelDataBase.__new__(ModelDataBase, '', '')
#          
#         #empty folder causes error        
#         e.tempdir = self.empty_folder
#         self.assertRaises(RuntimeError, lambda: ModelDataBase.check_already_build(e))   
#          
#         #tempdir contains full data and can therefore be read
#         e.tempdir = self.test_temp
#         self.assertTrue(ModelDataBase.check_already_build(e)) 
 
#     def test_read_db_already_build(self, *args):
#         '''if db is already build, it should be read and not build again'''
#         e = ModelDataBase.__new__(ModelDataBase, self.test_data, self.test_temp)
#         e.read_db = MagicMock()
#         e.build_db = MagicMock()
#         e.save_db = MagicMock()
#         e.__init__(self.test_data, self.test_temp)
#         e.read_db.assert_called_once_with()
#         e.build_db.assert_not_called()
          
#     def test_read_db_non_existent_tempdir(self, *args):
#         '''if tempdir is not existent, database should be build'''        
#         e = ModelDataBase.__new__(ModelDataBase, self.test_temp, self.nonexistent)
#         e.read_db = MagicMock()
#         e.build_db = MagicMock()
#         e.save_db = MagicMock()
#         e.__init__(self.test_temp, self.nonexistent)        
#         e.read_db.assert_not_called()
#         e.build_db.assert_called_once_with()

# obsolete, since the initializers are not called from the __init__ function of ModelDataBase anymore      
#     def test_non_existent_path(self):
#         '''if path to simulation data is not valid, an exeption is raised'''
#         self.assertRaises(RuntimeError, lambda: ModelDataBase(self.nonexistent, self.test_temp))
    
    @decorators.testlevel(1)    
    def test_canbeinstanciated(self):
        mdbtest = ModelDataBase(os.path.join(parent, 'test/data/test_temp'))

    @decorators.testlevel(3)    
    def test_no_empy_rows(self):
        e = ModelDataBase(os.path.join(parent, 'test/data', 'trash_it_now'))
        synapse_activation = e['synapse_activation']
        synapse_activation['isnan'] = synapse_activation.soma_distance.apply(lambda x: np.isnan(x))
        nr_nan_rows_synapse_activation = len(synapse_activation[synapse_activation.isnan == True])
        self.assertEqual(0, nr_nan_rows_synapse_activation)
        ##import shutil
        ##shutil.rmtree(os.path.join(parent, 'test/data', 'trash_it_now'))
        
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
         
         