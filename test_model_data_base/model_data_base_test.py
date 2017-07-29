from .context import *
from model_data_base.model_data_base import ModelDataBase, MdbException
import model_data_base.IO.LoaderDumper.to_pickle  as to_pickle
from . import decorators
import unittest
import os, shutil
import numpy as np
import tempfile

class Tests(unittest.TestCase):       
    def setUp(self):        
        self.mdb = ModelDataBase(test_mdb_folder)
        self.path_fresh_mdb = tempfile.mkdtemp()
        self.fresh_mdb = ModelDataBase(self.path_fresh_mdb)
        
    def tearDown(self):
        shutil.rmtree(self.path_fresh_mdb)
        
    def test_check_working_dir_clean_for_build_works_correctly(self):
        #can create database in empty folder
        testpath = tempfile.mkdtemp()
        ModelDataBase(testpath)
        
        #cannot create database if file is in folder
        with open(os.path.join(testpath, 'somefile'), 'w'):
            pass
        self.assertRaises(Exception, ModelDataBase(testpath))
        
        #can create database if folder can be created but does not exist
        shutil.rmtree(testpath)
        ModelDataBase(testpath)

        #cannot create database if subfolder is in folder
        shutil.rmtree(testpath)
        os.makedirs(os.path.join(testpath, 'somefolder'))
        self.assertRaises(MdbException, lambda: ModelDataBase(testpath))
        
        #tidy up
        shutil.rmtree(testpath)
        
    
    def test_mdb_does_not_permit_writes_if_readonly(self):
        mdb = ModelDataBase(self.path_fresh_mdb, readonly = True)
        def fun():
            mdb['test'] = 1
        self.assertRaises(MdbException, fun)
    
    def test_mdb_will_not_be_created_if_nocreate(self):
        testpath = tempfile.mkdtemp()
        self.assertRaises(MdbException, lambda: ModelDataBase(testpath, nocreate=True))
        ModelDataBase(self.path_fresh_mdb, nocreate = True)
        shutil.rmtree(testpath)
            
    def test_managed_folder_really_exists(self):
        self.fresh_mdb.create_managed_folder('asd')
        self.assertTrue(os.path.exists(self.fresh_mdb['asd']))
        
        #deleting the db entry deletes the folder
        folder_path = self.fresh_mdb['asd']
        del self.fresh_mdb['asd']
        self.assertFalse(os.path.exists(folder_path))
    
    def test_managed_folder_does_not_overwrite_existing_keys(self):
        self.fresh_mdb.create_managed_folder('asd')
        self.assertRaises(MdbException, lambda: self.fresh_mdb.create_managed_folder('asd'))
    
    def test_can_instantiate_sub_mdb(self):
        self.fresh_mdb.create_sub_mdb('test_sub_mdb')
        self.assertIsInstance(self.fresh_mdb['test_sub_mdb'], ModelDataBase)
    
    def test_sub_mdb_does_not_overwrite_existing_keys(self):
        self.fresh_mdb.setitem('asd', 1)
        self.assertRaises(MdbException, lambda: self.fresh_mdb.create_sub_mdb('asd'))
    
    def test_can_set_items_using_different_dumpers(self):
        self.fresh_mdb.setitem('test_self', 1, dumper = 'self')
        self.fresh_mdb.setitem('test_to_pickle', 1, dumper = to_pickle)
        self.assertEqual(self.fresh_mdb['test_self'], \
                         self.fresh_mdb['test_to_pickle'])
    
    def test_setitem_allows_replacing_an_existing_key_while_simulatneously_using_it(self):
        self.fresh_mdb['test'] = 1
        self.fresh_mdb['test'] = self.fresh_mdb['test']+1
        self.assertEqual(self.fresh_mdb['test'], 2)
        
    def test_maybe_calculate(self):
        pass

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
         
         