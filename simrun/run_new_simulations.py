'''
Created on Jan 28, 2013

ongoing activity L2 neuron model

@author: robert, arco
'''
from __future__ import absolute_import
from ._matplotlib_import import *
import sys
import time
import os, os.path
import glob
#for some reason, the neuron import works on the florida servers only works if tables was imported first
import tables
import neuron
import single_cell_parser as scp
import single_cell_parser.analyze as sca
import numpy as np
import logging
logger = logging.getLogger("ISF").getChild(__name__)

h = neuron.h
import dask
from .seed_manager import get_seed
from .utils import *
from data_base.dbopen import resolve_db_path
from biophysics_fitting.utils import execute_in_child_process
import socket

def _evoked_activity(
        cellParamName, 
        evokedUpParamName, 
        dirPrefix = '',
        seed = None, 
        nSweeps = 1000, 
        tStop = 345.0,
        tStim = 245.0, 
        scale_apical = None,
        cell_generator = None, 
        tar = False
        ):
    '''Run simulations of synaptic input patterns onto a biophysically detailed cell.

    This is the core method used throughout :py:mod:`simrun` to read in parameterfiles for the cell
    and network, set up a NEURON simulation environment, generate synaptic input patterns and saving the
    output data. This method should not be invoked directly, but is used by other methods in :py:mod:`simrun`,
    such as :py:meth:`~simrun.run_new_simulations`.
    
    The workflow of this method is as follows:

    1. Initialize the simulation
        1.1 Set a random seed. Used in the output directory name, and for generating network realizations and 
        network activity with :py:class:`~single_cell_parser.network.NetworkMapper`.
        1.2 Build the cell with biophysical properties.
        1.3 Set up the simulation with recording sites from the neuron parameters
    2. Run :paramref:`nSweeps` simulations using :py:meth:`~single_cell_parser.init_neuron_run`, 
    each time creating a new network embedding and sampling new activity using :py:meth:`~single_cell_parser.network.Network.create_saved_network2`.
    3. Parse and write out simulation data, including voltage traces from the soma 
    and additional recording sites defined in the neuron parameters.
    4. Finalize the simulation by removing the "_running" suffix from the dirname
    ":paramref:`dirPrefix`/results/%Y%M%D-%H%M_UID_running".
    
    Args:
        cellParamName (str): Path of the cell parameter file.
        evokedUpParamName (str): Path of the network parameter file.
        dirPrefix (str): Prefix of the output directory. The final directory name will be
            ":paramref:`dirPrefix`/results/%Y%M%D-%H%M_UID".
        seed (int): Random seed for the simulation.
        nSweps (int): Number of simulations to run with these parameters.
            Trial-to-trial variability is introduced by the random seed in terms of
            different network activity and connectivity realizations (see :py:meth:`~single_cell_parser.network.Network.created_saved_network2`).
        tStop (float): Duration of each simulation in ms.
        tStim (float): Time in ms at which the in-vivo evoked synaptic input should start.
        scale_apical (function): Function to scale the apical dendrite.
            Assumes the cell has an apical dendrite - see below.
        cell_generator (function): Function to generate the cell. If provided, the cell parameters
            provided by :paramref:`cellParamName` are ignored.
        tar (bool): If True, the output directory is compressed to a tarball after the simulation is finished.

    Attention:
     
        :paramref:`scale_apical` assumes that the cell has an apical dendrite:
        
        - It contains at least one section with the label "ApicalDendrite"
        - Such section exists at :paramref:`~dist` distance from the soma
        - The section has at least one child
            
        See :py:meth:`~get_inner_section_at_distance` for more information about which arguments can be used
        to define an apical dendrite.

    Returns:
        str: Path to the output directory containing the simulation results.
    '''
    # 1: Initialize the simulation ------------------------------
    # 1.1 Set seed
    assert seed is not None
    np.random.seed(seed)
    logger.info("seed: {}".format(seed))

    # Load in and parse parameter files
    neuronParameters = load_param_file_if_path_is_provided(cellParamName)
    evokedUpNWParameters = load_param_file_if_path_is_provided(
        evokedUpParamName)  ##sumatra function for reading in parameter file
    scp.load_NMODL_parameters(neuronParameters)
    scp.load_NMODL_parameters(evokedUpNWParameters)
    cellParam = neuronParameters.neuron
    paramEvokedUp = evokedUpNWParameters.network

    # 1.2 Build cell
    if cell_generator is None:
        cell = scp.create_cell(cellParam, scaleFunc=scale_apical)
    else:
        logger.warning(
            "Cell is generated by cell_generator. "
            "The the cell parameters provided by the cellParamName argument are ignored!")
        cell = cell_generator()

    # 1.3 Set up the simulation
    # Create output directory
    uniqueID = 'seed' + str(seed) + '_pid' + str(os.getpid())
    dirName = os.path.join(
        resolve_db_path(dirPrefix), 
        'results', 
        time.strftime('%Y%m%d-%H%M') + '_' + str(uniqueID) + '_running'
        )
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    with open(
        os.path.join(
            dirName, 
            'hostname_' + socket.gethostname()
            ), 'w') as f:
        pass

    # Simulation setup
    vTraces = []
    tTraces = []
    recordingSiteFiles = neuronParameters.sim.recordingSites
    recSiteManagers = []
    for recFile in recordingSiteFiles:
        recSiteManagers.append(sca.RecordingSiteManager(recFile, cell))

    tOffset = 0.0  # avoid numerical transients
    neuronParameters.sim.tStop = tStop
    dt = neuronParameters.sim.dt
    offsetBin = int(tOffset / dt + 0.5)

    # 2. Run simulations -------------------------------------------------
    nRun = 0
    while nRun < nSweeps:
        synParametersEvoked = paramEvokedUp

        # 2.1 Setup network from evoked network parameters
        startTime = time.time()
        evokedNW = scp.NetworkMapper(
            cell, 
            synParametersEvoked,
            neuronParameters.sim)
        logger.info('*' * 500)
        evokedNW.create_saved_network2()
        stopTime = time.time()
        setupdt = stopTime - startTime
        logger.info('Network setup time: {:.2f} s'.format(setupdt))

        synTypes = list(cell.synapses.keys())
        synTypes.sort()

        # 2.2 Activate synapses and track voltage.
        logger.info(
            'Testing evoked response properties run {:d} of {:d}'.format(nRun + 1, nSweeps))
        tVec = h.Vector()
        tVec.record(h._ref_t)
        startTime = time.time()
        scp.init_neuron_run(
            neuronParameters.sim,
            vardt=False)  # trigger the actual simulation
        stopTime = time.time()
        simdt = stopTime - startTime
        logger.info('NEURON runtime: {:.2f} s'.format(simdt))

        # 2.3 Extract simulation data to Python
        vmSoma = np.array(cell.soma.recVList[0])
        t = np.array(tVec)
        vTraces.append(np.array(vmSoma[offsetBin:])), tTraces.append(
            np.array(t[offsetBin:]))
        for RSManager in recSiteManagers:
            RSManager.update_recordings()

        # 2.4 Write out synaptic activations
        logger.info('Writing simulation results')
        fname = 'simulation'
        fname += '_run%04d' % nRun

        synName = dirName + '/' + fname + '_synapses.csv'
        logger.info('Computing active synapse properties')
        sca.compute_synapse_distances_times(
            synName, 
            cell, 
            t,
            synTypes)  # calls scp.write_synapse_activation_file
        preSynCellsName = dirName + '/' + fname + '_presynaptic_cells.csv'
        scp.write_presynaptic_spike_times(preSynCellsName, evokedNW.cells)

        # 2.5 Prepare next simulation run
        nRun += 1
        cell.re_init_cell()
        evokedNW.re_init_network()
    logger.info('-------------------------------')

    # 3. Save output data ---------------------------------------------------
    # 3.1 Parse and write out voltage traces
    vTraces = np.array(vTraces)
    dendTraces = []

    scp.write_all_traces(
        dirName + '/' + uniqueID + '_vm_all_traces.csv',
        t[offsetBin:], 
        vTraces)
    for RSManager in recSiteManagers:
        for recSite in RSManager.recordingSites:
            tmpTraces = []
            for vTrace in recSite.vRecordings:
                tmpTraces.append(vTrace[offsetBin:])
            recSiteName = dirName + '/' + uniqueID + '_' + recSite.label + '_vm_dend_traces.csv'
            scp.write_all_traces(recSiteName, t[offsetBin:], tmpTraces)
            dendTraces.append(tmpTraces)
    dendTraces = np.array(dendTraces)

    # 3.2 Write out simulation parameters
    logger.info('Writing simulation parameter files')
    neuronParameters.save(
        os.path.join(dirName, uniqueID + '_neuron_model.param'))
    evokedUpNWParameters.save(
        os.path.join(dirName, uniqueID + '_network_model.param'))
    
    # 4. Finalize simulation ---------------------------------------------------
    dirName_final = os.path.join(
        resolve_db_path(dirPrefix), 
        'results', 
        time.strftime('%Y%m%d-%H%M') + '_' + str(uniqueID)
        )
    os.rename(dirName, dirName_final)
    if tar:
        tar_folder(dirName_final + '.running', True)
    return dirName_final

def run_new_simulations(
        cellParamName, 
        evokedUpParamName, 
        dirPrefix = '',
        nSweeps = 1000, 
        nprocs = 40, 
        tStop = 345, 
        silent = True,
        scale_apical = None,
        cell_generator = None,
        child_process = False,
        tar = False
        ):
    '''Run new simulations of synaptic input patterns onto a biophysically detailed cell.
    
    Generates :paramref:`nSweeps`*:paramref:`nprocs` synapse activation files and saves them in
    the folder :paramref:`dirPrefix`/results/%Y%M%D-%H%M_UID. 
    Does not execute the simulations directly, but creates a list of delayed objects for each process, 
    which can be computed with an arbitrary dask scheduler. 
    For each process, a new seed is generated using :py:mod:`~simrun.seed_manager`.
    
    Args:
        cellParamName (str): 
            Path to a cell parameter file (e.g. getting_started/example_data/biophysical_constraints/*.param), 
            containing information about the neuron morphology (link to a .hoc file) and biophysical properties.
            See :py:meth:`~single_cell_parser.create_cell` for more information on the structure and contents of this filetype
        evokedUpParamName (str): 
            Path to network parameter file (e.g. getting_started/example_data/functional_constraints/network.param),
            containing information on synapse and network parameters per cell type. See :py:meth:`~singlecell_input_mapper.evoked_network_param_from_template.create_network_parameter`
            for more information on the structure and contents of this filetype
        nSweeps (int): 
            number of synapse activations per process
        nprocs (int): 
            number of independent processes
        tStop (float): 
            time in ms at which the synaptic input should stop.
        
    Returns: 
        list: 
            A list of dask.delayed.Delayed objects containing the simulation instructions.
            The list can be computed with an arbitrary dask scheduler.
            Each element in the list corresponds to one process, so the list has length :paramref:`nprocs`.
    '''

    myfun = lambda seed: _evoked_activity(
        cellParamName, 
        evokedUpParamName,
        dirPrefix = dirPrefix, 
        seed = seed, 
        nSweeps = nSweeps,
        tStop = tStop, 
        scale_apical = scale_apical,
        cell_generator = cell_generator,
        tar = tar
        )
    if silent:
        myfun = silence_stdout(myfun)

    if child_process:
        myfun = execute_in_child_process(myfun)

    d = [dask.delayed(myfun)(get_seed()) for i in range(nprocs)]
    return d  # dask.delayed(lambda *args: args)(d) #return single delayed object, that computes everything
