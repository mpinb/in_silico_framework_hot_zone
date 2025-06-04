import os
import neuron
import socket
import distributed
import pytest
import getting_started

h = neuron.h
import single_cell_parser as scp
from data_base.utils import silence_stdout
import mechanisms.l5pt
from tests.context import TEST_DATA_FOLDER

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
        '86_C2_center.param')
    cell_param = scp.build_parameters(
        cell_param)  # this is the main method to load in parameterfiles
    # load scaled hoc morphology
    cell_param.neuron.filename = os.path.join(
        TEST_DATA_FOLDER,
        'anatomical_constraints', 
        '86_C2_center_scaled_diameters.hoc')
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

    rangevars = rangevars or []
    
    neup = scp.ParameterSet(getting_started.neuronParam)
    netp = scp.ParameterSet(getting_started.networkParam)
    
    cell = scp.create_cell(neup.neuron)
    evokedNW = scp.NetworkMapper(cell, netp.network, neup.sim)
    evokedNW.create_saved_network2()
    
    neup.sim.tStop = 10

    for rv in rangevars:
        cell.record_range_var(rv)
        
    scp.init_neuron_run(neup.sim, vardt=False)
    
    evokedNW.re_init_network()

    return cell

def is_port_open(host, port):
    try:
        port = int(port)
    except ValueError as e:
        raise ValueError("Port must be an integer or integer-convertible") from e
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # 1 second timeout
        result = sock.connect_ex((host, port))
        return result == 0


def client(port):
    if is_port_open("localhost", port):
        host = "localhost"
    else:
        host = socket.gethostbyname(socket.gethostname())

    c = distributed.Client(f"{host}:{port}")
    return c

