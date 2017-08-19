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
    
    def test_setitem_allows_replacing_an_existing_key_while_simultaneously_using_it(self):
        self.fresh_mdb['test'] = 1
        self.fresh_mdb['test'] = self.fresh_mdb['test']+1
        self.assertEqual(self.fresh_mdb['test'], 2)