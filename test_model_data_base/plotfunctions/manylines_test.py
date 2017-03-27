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
        if savefigs: 
            print """Testing manyilines plots. Output files are saved in %s. 
                Please make sure that they display the same data."""
        
    def test_manylines_no_group(self):
        df = self.df.drop('attribute', axis = 1)
        ddf = dd.from_pandas(df, npartitions = 3)
        fig = plt.figure()
        manylines(df, axis = [1, 10, 1, 10], fig = fig)
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_no_group_pandas.png'))
        fig = plt.figure()
        manylines(ddf, axis = [1, 10, 1, 10], fig = fig)
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_no_group_dask.png'))
         
    def test_manylines_grouped(self):
        df = self.df
        ddf = dd.from_pandas(df, npartitions = 3)
        fig = plt.figure()
        manylines(df, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, fig = fig)
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_grouped_pandas.png'))
        fig = plt.figure()        
        manylines(ddf, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, fig = fig)
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_grouped_dask.png'))

    def test_manylines_no_group_returnPixelObject(self):
        df = self.df.drop('attribute', axis = 1)
        po = manylines(df, axis = [1, 10, 1, 10], returnPixelObject = True)
        self.assertIsInstance(po, PixelObject)
        fig = plt.figure()
        show_pixel_object(po, fig = fig)
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_no_group_po_pandas.png'))
        
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
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_grouped_po_pandas.png'))
        po = manylines(ddf, axis = [1, 10, 1, 10], \
                        groupby_attribute = 'attribute', \
                        colormap = self.colormap, \
                        returnPixelObject = True)
        self.assertIsInstance(po, PixelObject)
        fig = plt.figure()
        show_pixel_object(po, fig = fig)
        if savefigs: fig.savefig(os.path.join(files_generated_by_tests, 'manylines_grouped_po_dask.png'))
         
                     