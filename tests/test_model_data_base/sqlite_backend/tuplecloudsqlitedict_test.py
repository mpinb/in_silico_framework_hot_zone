import unittest
import shutil
import os
import numpy as np
import dask
import distributed
from ..context import *
from model_data_base.sqlite_backend.tuplecloudsqlitedict import SqliteDict, check_key

@dask.delayed
def write_data_to_dict(path, key):
    dict_ = SqliteDict(path, autocommit = True)
    data = np.random.rand(1000,1000)
    dict_[key] = data
            
            
class Tests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tempdir, 'tuplecloudsql_test.db')
        self.db = SqliteDict(self.path, autocommit = True, flag = 'c')
        
    def tearDown(self):
        self.db.close()
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
            
    def test_check_key(self):
        self.assertRaises(ValueError, lambda: check_key(1))
        check_key('1')
        self.assertRaises(ValueError, lambda: check_key((1,)))
        check_key(('1',)) 
        self.assertRaises(ValueError, lambda: check_key('@'))
        self.assertRaises(ValueError, lambda: check_key(('@asd', 'asd')))      
    
    def test_str_values_can_be_assigned(self):
        db = self.db
        db['test'] = 'test'
        self.assertEqual(db['test'], 'test')
    
    def test_tuple_values_can_be_assigned(self):
        db = self.db
        db[('test',)] = 'test'
        self.assertEqual(db[('test',)], 'test')  
        db[('test','abc')] = 'test2'
        self.assertEqual(db[('test','abc')], 'test2')  
        
    def test_pixelObject_can_be_assigned(self):
        db = self.db
        #plot figure and convert it to PixelObject
        import matplotlib.pyplot as plt
        from visualize._figure_array_converter import PixelObject
        fig = plt.figure()
        fig.add_subplot(111).plot([1,5,3,4])
        po = PixelObject([0, 10, 0, 10], fig = fig)
         
        #save and reload PixelObject
        db[('test', 'myPixelObject')] = po
        po_reconstructed = db[('test', 'myPixelObject')]
            
        
        
        
