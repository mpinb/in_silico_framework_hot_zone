import os
import neuron

h = neuron.h
from Interface import scp
from data_base.utils import silence_stdout
import mechanisms.l5pt
from tests.context import TEST_DATA_FOLDER

def get_n_decimals(platform, neuron_version):
    if platform == "linux":
        assert neuron_version.startswith("7.8.2"), \
            "For exact reproducability, NEURON 7.8.2 is required on linux"
    if neuron_version.startswith("7.8.2"):
        return 6  # For the original NEURON version, we want exact reproducibility
    else:
        return 2  # Otherwise, we allow for some numerical differences
    
def get_rel_tolerance(platform, neuron_version):
    if platform == "linux":
        assert neuron_version.startswith("7.8.2"), \
            "For exact reproducability, NEURON 7.8.2 is required on linux"
    if neuron_version.startswith("7.8.2"):
        return 1e-6  # For the original NEURON version, we want exact reproducibility
    else:
        return 1e-2  # Otherwise, we allow for 1% numerical differences

def setup_current_injection_experiment(
        rangevars=None
        ):
    """
    Sets up a current injection experiment of some :ref:`hoc_file_format` and .param file.
    The following parameters define the experiment:

    Returns:
        cell: a cell object that contains the simulation.
    """
    rangevars = rangevars or []
    cell_param = os.path.join(
        TEST_DATA_FOLDER,
        'biophysical_constraints', 
        '86_CDK_20041214_BAC_run5_soma_Hay2013_C2center_apic_rec.param')
    cell_param = scp.build_parameters(
        cell_param)  # this is the main method to load in parameterfiles
    # load scaled hoc morphology
    cell_param.neuron.filename = os.path.join(
        TEST_DATA_FOLDER,
        'anatomical_constraints', 
        '86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2center_scaled_diameters.hoc')
    with silence_stdout:
        cell = scp.create_cell(cell_param.neuron)

    iclamp = h.IClamp(0.5, sec=cell.soma)
    iclamp.delay = 150  # give the cell time to reach steady state
    iclamp.dur = 5  # 5ms rectangular pulse
    iclamp.amp = 1.9  # 1.9 ?? todo ampere
    for rv in rangevars:
        cell.record_range_var(rv)

    scp.init_neuron_run(cell_param.sim, vardt=True)  # run the simulation

    return cell

def setup_synapse_activation_experiment(
        rangevars=None
        ):
    """
    Sets up a current injection experiment of some :ref:`hoc_file_format` and .param file.
    The following parameters define the experiment:

    Returns:
        cell: a cell object that contains the simulation.
    """
    
    import getting_started
    import single_cell_parser as scp

    rangevars = rangevars or []
    
    neup = scp.NTParameterSet(getting_started.neuronParam)
    netp = scp.NTParameterSet(getting_started.networkParam)
    
    cell = scp.create_cell(neup.neuron)
    evokedNW = scp.NetworkMapper(cell, netp.network, neup.sim)
    evokedNW.create_saved_network2()
    
    neup.sim.tStop = 10

    for rv in rangevars:
        cell.record_range_var(rv)
        
    scp.init_neuron_run(neup.sim, vardt=False)
    
    evokedNW.re_init_network()

    return cell