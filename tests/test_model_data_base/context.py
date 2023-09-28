from __future__ import absolute_import
import os, sys, shutil, tempfile
import distributed
import pytest
parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent)
import distributed
from model_data_base import *
import getting_started
import mechanisms
# set up paths
test_data_folder = os.path.join(getting_started.parent, \
                              'example_simulation_data', \
                              'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/')

# cell_param_path = os.path.join(parent, 'test_model_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/20240_neuron_model.param')
# assert(os.path.exists(cell_param_path))
# network_param_path = os.path.join(parent, 'test_model_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/20240_network_model.param')
# assert(os.path.exists(network_param_path))
# test_data_folder = os.path.join(parent, 'test_model_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/')

class FreshlyInitializedMdb(object):
    '''context manager that provides a freshly initalized mdb for 
    testing purposes'''
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        self.mdb = model_data_base.ModelDataBase(self.path)
        #self.mdb.settings.show_computation_progress = False
        from model_data_base.mdb_initializers.load_simrun_general import init
        from model_data_base.utils import silence_stdout
        with silence_stdout:
            init(self.mdb, test_data_folder, client = distributed.client_object_duck_typed, 
                 rewrite_in_optimized_format=False, 
                 parameterfiles=False,
                 dendritic_voltage_traces=False)
       
        return self.mdb
    
    def __exit__(self, *args, **kwargs):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

