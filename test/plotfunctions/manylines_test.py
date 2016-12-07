from ..context import *
from model_data_base.plotfunctions.manylines import *
import unittest
import pandas as pd
import dask.dataframe as dd
from model_data_base.plotfunctions._figure_array_converter import PixelObject, show_pixel_object 

savefigs = True

class Tests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'1': [1,2,3,4,5], \
                           '2': [2,1,6,3,4], \
                           '3': [7,3,4,1,2], \
                           'attribute': ['a', 'a', 'a', 'b', 'b']})
        
        self.colormap = dict(a='r', b='b')
        
    def test_manylines_no_group(self):
        df = self.df.drop('attribute', axis = 1)
        ddf = dd.from_pandas(df, npartitions = 3)
        fig = plt.figure()
        manylines(df, axis = [1, 10, 1, 10], fig = fig)
        if savefigs: fig.savefig('test1.png')
        fig = plt.figure()
        manylines(ddf, axis = [1, 10, 1, 10], fig = fig)
        if savefigs: fig.savefig('test2.png')
         
    def test_manylines_grouped(self):
        df = self.df
        ddf = dd.from_pandas(df, npartitions = 3)
        fig = plt.figure()
        manylines(df, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, fig = fig)
        if savefigs: fig.savefig('test3.png')
        fig = plt.figure()        
        manylines(ddf, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, fig = fig)
        if savefigs: fig.savefig('test4.png')

    def test_manylines_no_group_returnPixelObject(self):
        df = self.df.drop('attribute', axis = 1)
        ddf = dd.from_pandas(df, npartitions = 3)
        po = manylines(df, axis = [1, 10, 1, 10], returnPixelObject = True)
        self.assertIsInstance(po, PixelObject)
        fig = plt.figure()
        show_pixel_object(po, fig = fig)
        if savefigs: fig.savefig('test1_returnPixelObject.png')
        
    def test_manylines_grouped_returnPixelObject(self):
        df = self.df
        ddf = dd.from_pandas(df, npartitions = 3)
        po = manylines(df, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, \
                        returnPixelObject = True)
        self.assertIsInstance(po, PixelObject)        
        fig = plt.figure()
        show_pixel_object(po, fig = fig)
        if savefigs: fig.savefig('test3_returnPixelObject.png')
        po = manylines(ddf, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, \
                        returnPixelObject = True)
        self.assertIsInstance(po, PixelObject)
        fig = plt.figure()
        show_pixel_object(po, fig = fig)
        if savefigs: fig.savefig('test4_returnPixelObject.png')
         
                     