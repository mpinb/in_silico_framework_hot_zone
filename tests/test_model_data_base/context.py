from __future__ import absolute_import
import os, sys, shutil, tempfile
import distributed
import pytest

parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent)
import distributed
from isf_data_base import *
import getting_started
import mechanisms
# set up paths
test_data_folder = os.path.join(getting_started.parent, \
                              'example_simulation_data', \
                              'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/')

# cell_param_path = os.path.join(parent, 'test_isf_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/20240_neuron_model.param')
# assert os.path.exists(cell_param_path)
# network_param_path = os.path.join(parent, 'test_isf_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/20240_network_model.param')
# assert os.path.exists(network_param_path)
# test_data_folder = os.path.join(parent, 'test_isf_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/')


class FreshlyInitializedMdb(object):
    '''context manager that provides a freshly initalized db for 
    testing purposes

    It is recommended to use the pytest fixture fresh_db instead of this class, except when the location of the db is of importance. This is the case in tests:
    - tests/test_isf_data_base/Model_data_base_test.py::test_compare_old_db_with_freshly_initialized_one
    '''

    def __init__(self, *args, **kwargs):
        self.path = None
        self.db = None

    def __call__(self, client):
        self.path = tempfile.mkdtemp()
        self.db = isf_data_base.DataBase(self.path)
        #self.db.settings.show_computation_progress = False
        from isf_data_base.db_initializers.load_simrun_general import init
        from isf_data_base.utils import silence_stdout
        with silence_stdout:
            init(self.db,
                 test_data_folder,
                 client=client,
                 rewrite_in_optimized_format=False,
                 parameterfiles=False,
                 dendritic_voltage_traces=False)

    def __enter__(self):
        return self.db

    def __exit__(self, *args, **kwargs):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
