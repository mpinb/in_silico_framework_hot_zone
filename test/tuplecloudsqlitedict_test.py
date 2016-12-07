import unittest
import shutil
import os
from context import *
from model_data_base.tuplecloudsqlitedict import SqliteDict


class Tests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(parent, 'test', 'tuplecloudsql_test.db')
    def test_can_be_instanciated(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        db = SqliteDict(self.path, autocommit = True)
    
    def test_str_values_can_be_assigned(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        db = SqliteDict(self.path, autocommit = True)
        db['test'] = 'test'
        self.assertEqual(db['test'], 'test')
    
    def test_tuple_values_can_be_assigned(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        db = SqliteDict(self.path, autocommit = True)
        db[('test',)] = 'test'
        self.assertEqual(db[('test',)], 'test')  
        db[('test','abc')] = 'test2'
        self.assertEqual(db[('test','abc')], 'test2')  
        
    def test_pixelObject_can_be_assigned(self):
        #setup db
        if os.path.exists(self.path):
            os.remove(self.path)
        db = SqliteDict(self.path, autocommit = True)
        
        #plot figure and convert it to PixelObject
        import matplotlib.pyplot as plt
        from model_data_base.plotfunctions._figure_array_converter import PixelObject
        fig = plt.figure()
        fig.add_subplot(111).plot([1,5,3,4])
        po = PixelObject([0, 10, 0, 10], fig = fig)
        
        #save and reload PixelObject
        db[('test', 'myPixelObject')] = po
        po_reconstructed = db[('test', 'myPixelObject')]
        
