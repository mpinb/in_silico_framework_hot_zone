from __future__ import absolute_import
import unittest
import numpy as np
import neuron
h = neuron.h
import pickle
from single_cell_parser.serialize_cell import *
from .context import *
from test import setup_current_injection_experiment


class Tests(unittest.TestCase): 
    def setUp(self):
        self.cell = setup_current_injection_experiment()
    
    def test_can_be_pickled(self):
        silent = cell_to_serializable_object(self.cell)
        pickle.dumps(silent)
        
    def test_values_are_the_same_after_reload(self):
        silent = cell_to_serializable_object(self.cell)
        cell2 = restore_cell_from_serializable_object(silent)
        
        np.testing.assert_array_equal(np.array(self.cell.tVec), cell2.tVec)
        np.testing.assert_array_equal(np.array(self.cell.soma.recVList[0]), \
                                      cell2.soma.recVList[0])        