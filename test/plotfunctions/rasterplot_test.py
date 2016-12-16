from ..context import *
from model_data_base.plotfunctions.rasterplot import *
import unittest
import dask.dataframe as dd
import pandas as pd
from .. import decorators
from model_data_base.model_data_base import ModelDataBase 

class Tests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'1': [1,2,3,4,5], \
                           '2': [2,1,6,3,4], \
                           '3': [7,3,4,1,2], \
                           'attribute': ['a', 'a', 'a', 'b', 'b']})
        
        self.colormap = dict(a='r', b='b')
    
    def test_pandas(self):
        fig = rasterplot(self.df, tlim = (0,350))
    
    def test_dask(self):
        ddf = dd.from_pandas(self.df, npartitions = 2)
        fig = rasterplot(self.df, tlim = (0,350))

#     @decorators.testlevel(1)        
#     def test_real_data(self):
#         mdb = ModelDataBase(os.path.join(parent, 'trash_it_now'))
#         x = mdb['spike_times']
#         if not isinstance(x, dd.DataFrame):
#             raise ValueError("This test requires mdb['spike_times'] to be an instance of dask.dataframe.Dataframe")
#         fig = rasterplot(mdb['spike_times'], tlim = (0,350))
    
    def test_can_be_called_with_figures_and_axes(self):
        from  matplotlib.figure import Figure
        from matplotlib.axes import Axes
        fig = plt.figure(figsize = (15,3))
        ax = fig.add_subplot(1,1,1)
        self.assertIsInstance(rasterplot(self.df, tlim = (0,350)), Figure)
        self.assertIs(rasterplot(self.df, tlim = (0,350), fig = fig), fig)
        self.assertIs(rasterplot(self.df, tlim = (0,350), fig = ax), ax)  
    
    
# class Tests(unittest.TestCase):
#     def setUp(self):
#         self.df = pd.DataFrame({'1': [1,2,3,4,5], \
#                            '2': [2,1,6,3,4], \
#                            '3': [7,3,4,1,2], \
#                            'attribute': ['a', 'a', 'a', 'b', 'b']})
#         
#         self.colormap = dict(a='r', b='b')
#         
#     def test_manylines_no_group(self):
#         df = self.df.drop('attribute', axis = 1)
#         ddf = dd.from_pandas(df, npartitions = 3)
#         fig = manylines(df, axis = [0, 10, 0, 10])
#         if savefigs: fig.savefig('test1.png')
#         fig = manylines(ddf, axis = [0, 10, 0, 10])
#         if savefigs: fig.savefig('test2.png')
#         
#     def test_manylines_grouped(self):
#         df = self.df
#         ddf = dd.from_pandas(df, npartitions = 3)
#         fig = manylines(df, axis = [0, 10, 0, 10], \
#                         groupby_attribute = 'attribute', \
#                         colormap = self.colormap)
#         if savefigs: fig.savefig('test3.png')
#         fig = manylines(ddf, axis = [0, 10, 0, 10], \
#                         groupby_attribute = 'attribute', \
#                         colormap = self.colormap)
#         if savefigs: fig.savefig('test4.png')
# 
#             