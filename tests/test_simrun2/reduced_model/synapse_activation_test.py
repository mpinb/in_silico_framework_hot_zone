from __future__ import absolute_import
#from ..context import *
from simrun2.reduced_model.synapse_activation import get_poisson_realizations_from_expectancy_values
import unittest
import numpy as np
import Interface as I

class TestSynapseActivation(unittest.TestCase): 
    def test_get_poisson_realizations_from_expectancy_values(self):
        expectancy = [0,1,2,3,4]
        realization = get_poisson_realizations_from_expectancy_values(expectancy, nSweeps = 5000)
        np.testing.assert_almost_equal(expectancy, realization.mean(axis = 0), 1)