from .context import *
from model_data_base.utils import *
from . import decorators
import unittest
import numpy as np
from pandas.util.testing import assert_frame_equal

class Test(unittest.TestCase):
    def test_pandas_to_array(self):
        pdf = pd.Series({'x_1_y_1': 10, 'x_2_y_1': 15, 'x_3_y_1': 7,'x_1_y_2': 2, 'x_2_y_2': 0, 'x_3_y_2': -1}).to_frame(name = 'bla')

        pdf2 = pandas_to_array(pdf, lambda index, values: index.split('_')[1], \
                         lambda index, values: index.split('_')[-1], \
                         lambda index, values: values.bla)
        
        pdf3 = pd.DataFrame({'1': [10,2], '2': [15,0], '3': [7,-1], 'index': ['1','2']}).set_index('index')
        pdf3.index.name = None
        assert_frame_equal(pdf2, pdf3)