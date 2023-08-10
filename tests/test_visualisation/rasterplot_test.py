from .context import *
from visualize.rasterplot import *
import unittest
import dask.dataframe as dd
import pandas as pd
from . import decorators
from model_data_base.model_data_base import ModelDataBase 

class TestRasterplot(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'1': [1,2,3,4,5], \
                           '2': [2,1,6,3,4], \
                           '3': [7,3,4,1,2], \
                           'attribute': ['a', 'a', 'a', 'b', 'b']})

        self.colormap = dict(a='r', b='b')

    def test_pandas(self):
        fig = rasterplot(self.df, tlim = (0,350))
        plt.close()

    def test_dask(self):
        ddf = dd.from_pandas(self.df, npartitions = 2)
        fig = rasterplot(self.df, tlim = (0,350))
        plt.close()

    def test_can_be_called_with_figures_and_axes(self):
        from  matplotlib.figure import Figure
        from matplotlib.axes import Axes
        fig = plt.figure(figsize = (15,3))
        ax = fig.add_subplot(1,1,1)
        self.assertIsInstance(rasterplot(self.df, tlim = (0,350)), Figure)
        self.assertIs(rasterplot(self.df, tlim = (0,350), fig = fig), fig)
        self.assertIs(rasterplot(self.df, tlim = (0,350), fig = ax), ax)
        plt.close()
