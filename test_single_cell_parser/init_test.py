from __future__ import absolute_import
import single_cell_parser as scp
from getting_started import getting_started_dir # path to getting started folder
from model_data_base.utils import fancy_dict_compare
import os, unittest
from .context import *


class Tests(unittest.TestCase): 
    def setUp(self):
        self.cell_param = os.path.join(getting_started_dir, \
                            'biophysical_constraints', \
                            '86_CDK_20041214_BAC_run5_soma_Hay2013_C2center_apic_rec.param')
        self.network_param = os.path.join(getting_started_dir, \
                            'functional_constraints', \
                            'network.param')
        
        assert(os.path.exists(self.cell_param))
        assert(os.path.exists(self.network_param))
    
    def test_fast_and_slow_mode_of_build_parameters_gives_same_results(self):
        bp = scp.build_parameters
        comp = fancy_dict_compare(bp(self.cell_param, fast_but_security_risk = True), \
                                  bp(self.cell_param, fast_but_security_risk = False))
        self.assertEqual(comp, '')
        
        comp = fancy_dict_compare(bp(self.network_param, fast_but_security_risk = True), \
                                  bp(self.network_param, fast_but_security_risk = False))
        self.assertEqual(comp, '')        
        
        
        