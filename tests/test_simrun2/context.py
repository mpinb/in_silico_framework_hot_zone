from __future__ import absolute_import
import os
from tests.context import TEST_DATA_FOLDER, TEST_SIMULATION_DATA_FOLDER

cellParamName = os.path.join(
    TEST_DATA_FOLDER,
    'biophysical_constraints',
    '86_CDK_20041214_BAC_run5_soma_Hay2013_C2center_apic_rec.param')
networkName = os.path.join(
    TEST_DATA_FOLDER,
    'functional_constraints', 
    'network.param')
example_path = os.path.join(
    TEST_SIMULATION_DATA_FOLDER, 
    '20150815-1530_20240', 
    'simulation_run0000_synapses.csv')