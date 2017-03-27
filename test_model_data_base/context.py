from __future__ import absolute_import
import os, sys, shutil
parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent)

from model_data_base import *

import model_data_base.mdb_initializers.load_simrun_general
from model_data_base.model_data_base import ModelDataBase


# set up paths
cell_param_path = os.path.join(parent, 'test_model_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/20240_neuron_model.param')
assert(os.path.exists(cell_param_path))
network_param_path = os.path.join(parent, 'test_model_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/20240_network_model.param')
assert(os.path.exists(network_param_path))
test_data_folder = os.path.join(parent, 'test_model_data_base/data/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/')
assert(os.path.exists(test_data_folder))
test_mdb_folder = os.path.join(parent, 'test_model_data_base/data/test_mdb')
files_generated_by_tests = os.path.join(parent, 'test_model_data_base/data/files_generated_by_tests')

