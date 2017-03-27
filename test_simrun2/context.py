from __future__ import absolute_import
import os
import sys
parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent)

cell_param_path = os.path.join(parent, 'test_model_data_base/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/example_parameter_files/20150815-1530_20240/20240_neuron_model.param')
assert(os.path.exists(cell_param_path))
network_param_path = os.path.join(parent, 'test_model_data_base/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/example_parameter_files/20240_network_model.param')
assert(os.path.exists(network_param_path))
test_data_path = os.path.join(parent, 'test_model_data_base/test_data/C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/20150815-1530_20240/example_synapse_activation_file')
assert(os.path.exists(test_data_path))
 