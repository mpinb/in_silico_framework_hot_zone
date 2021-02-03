'''
Created on Jan 28, 2013

ongoing activity L2 neuron model

@author: robert, arco
'''
import six
if six.PY2:
    from __future__ import absolute_import
from ._matplotlib_import import * 
import sys
import time
import os, os.path
import glob
import neuron
import single_cell_parser as scp
import single_cell_analyzer as sca
import numpy as np
h = neuron.h
import dask
import pandas as pd
from .utils import *
    
def _evoked_activity(cellParamName, evokedUpParamName, synapse_activation_files, \
                     dirPrefix = '', tStop = 345.0, scale_apical = scale_apical, post_hook = {}, \
                     auto_organize_results_folder = True):
    '''
    pre-stimulus ongoing activity
    and evoked activity
    (
    sim name: str, describing the simulation
    cellParamName: str, Path to cell parameter file, containing information about:
        - info: autor, date, name
        - NMODL_mechanisms: path to NEURON mechanisms
        - neuron: 
            -path to hoc-file
            - per subcellular compartment (Soma, AIS, ...):
                - electrical properties
                - mechanisms
    evokedUpParamName: str, Path to network parameter file, containing information about:
                            - autor, name, date
                            - for each cell-type: 
                                synapse: release probability, path to distribution file, receptor and associated parameters
                                connectionFile: path to connection file
                                cell number
                                celltype: pointcell, spiketrain
    '''

    if len(synapse_activation_files) == 0:
        print('warning: empty_simulation')
        return
    
    if isinstance(cellParamName, str):
        neuronParameters = scp.build_parameters(cellParamName)
    else:
        neuronParameters = cellParamName
    
    evokedUpNWParameters = scp.build_parameters(evokedUpParamName) ##sumatra function for reading in parameter file
    scp.load_NMODL_parameters(neuronParameters)
    scp.load_NMODL_parameters(evokedUpNWParameters)
    cellParam = neuronParameters.neuron
    paramEvokedUp = evokedUpNWParameters.network

    cell = scp.create_cell(cellParam, scaleFunc=scale_apical)
            
    uniqueID = str(os.getpid())
    if auto_organize_results_folder:
        dirName = os.path.join(dirPrefix, 'results', \
                           time.strftime('%Y%m%d-%H%M') + '_' + str(uniqueID))
    else:
        dirName = dirPrefix
        
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
    ret_df = pd.DataFrame(dict(synapse_file = synapse_activation_files, \
                      number = ['%07d' % n for n in range(len(synapse_activation_files))]))
    ret_df.to_csv(os.path.join(dirName, 'synapse_file_paths.csv'), index = False)

    vTraces = []
    tTraces = []
    recordingSiteFiles = neuronParameters.sim.recordingSites
    recSiteManagers = []
    for recFile in recordingSiteFiles:
        recSiteManagers.append(sca.RecordingSiteManager(recFile, cell))
    
    tOffset = 0.0 # avoid numerical transients
    neuronParameters.sim.tStop = tStop
    dt = neuronParameters.sim.dt
    offsetBin = int(tOffset/dt + 0.5)
    
    nRun = 0
    
    if post_hook: ##
        post_hook_list = [] ##
        
    for synfile in synapse_activation_files:
        synParametersEvoked = paramEvokedUp
        
        startTime = time.time()
        evokedNW = scp.NetworkMapper(cell, synParametersEvoked, neuronParameters.sim)
#        evokedNW.create_saved_network2()
        evokedNW.reconnect_saved_synapses(synfile)
        
        stopTime = time.time()
        setupdt = stopTime - startTime
        print('Network setup time: {:.2f} s'.format(setupdt))
                
        synTypes = list(cell.synapses.keys())
        synTypes.sort()
        
        print('Testing evoked response properties run {:d} of {:d}'.format(nRun+1, len(synapse_activation_files)))
        tVec = h.Vector()
        tVec.record(h._ref_t)
        startTime = time.time()
        scp.init_neuron_run(neuronParameters.sim, vardt=False) #trigger the actual simulation
        stopTime = time.time()
        simdt = stopTime - startTime
        print('NEURON runtime: {:.2f} s'.format(simdt))
        
        vmSoma = np.array(cell.soma.recVList[0])
        t = np.array(tVec)
        cell.t = t ##
        vTraces.append(np.array(vmSoma[offsetBin:])), tTraces.append(np.array(t[offsetBin:]))
        for RSManager in recSiteManagers:
            RSManager.update_recordings()
        
        print('writing simulation results')
        fname = 'simulation'
        fname += '_run%07d' % nRun
        
        synName = dirName + '/' + fname + '_synapses.csv'
        print('computing active synapse properties')
        sca.compute_synapse_distances_times(synName, cell, t, synTypes) #calls scp.write_synapse_activation_file
        preSynCellsName = dirName + '/' + fname + '_presynaptic_cells.csv'
        scp.write_presynaptic_spike_times(preSynCellsName, evokedNW.cells)
        
        nRun += 1
        
        if post_hook: ##
            dummy = {}
            for name in post_hook:
                dummy[name] = post_hook[name](cell)
            post_hook_list.append(dummy)
        
        cell.re_init_cell()
        evokedNW.re_init_network()

        print('-------------------------------')
    
    vTraces = np.array(vTraces)
    dendTraces = []
    scp.write_all_traces(dirName+'/'+uniqueID+'_vm_all_traces.csv', t[offsetBin:], vTraces)
    for RSManager in recSiteManagers:
        for recSite in RSManager.recordingSites:
            tmpTraces = []
            for vTrace in recSite.vRecordings:
                tmpTraces.append(vTrace[offsetBin:])
            recSiteName = dirName +'/' + uniqueID + '_' + recSite.label + '_vm_dend_traces.csv'
            scp.write_all_traces(recSiteName, t[offsetBin:], tmpTraces)
            dendTraces.append(tmpTraces)
    dendTraces = np.array(dendTraces)
    
    print('writing simulation parameter files')
    neuronParameters.save(os.path.join(dirName, uniqueID + '_neuron_model.param'))
    evokedUpNWParameters.save(os.path.join(dirName, uniqueID+ '_network_model.param'))
    
    print('writing list of synapse files')
    if post_hook: ##
        return ret_df, dirName, post_hook_list
    else:
        return ret_df, dirName

def run_existing_synapse_activations(cellParamName, evokedUpParamName, synapseActivation, simName = '', dirPrefix = '', \
                                 nprocs = 40, tStop = 345, silent = True, \
                                 scale_apical = scale_apical, post_hook = {}, auto_organize_results_folder = True):
    '''Generates nSweeps*nprocs synapse activation files and puts them in
    the folder dirPrefix/results/simName. Returns delayed object, which can
    be computed with an arbitrary dask scheduler. For each process, a new
    seed is generated using the seed generator.
    
    Parameters:
        cellParamName: str, Path to cell parameter file, containing information about:
            - info: autor, date, name
            - NMODL_mechanisms: path to NEURON mechanisms
            - neuron: 
                -path to hoc-file
                - per subcellular compartment (Soma, AIS, ...):
                    - electrical properties
                    - mechanisms
        evokedUpParamName: str, Path to network parameter file, containing information about:
                                - autor, name, date
                                - for each cell-type: 
                                    synapse: release probability, path to distribution file, receptor and associated parameters
                                    connectionFile: path to connection file
                                    cell number
                                    celltype: pointcell, spiketrain
        synapseActivationGlob: List of paths to synapse activation files or globstring
        nSweeps: number of synapse activations per process
        nprocs: number of independent processes
        tStop: time in ms at which the synaptic input should stop.
        
    Returns: Delayed object. Can be computed with arbitrary scheduler.'''
    
    if not auto_organize_results_folder:
        if not nprocs == 1:
            raise NotImplementedError()
    if isinstance(synapseActivation, str):
        synapseActivation = glob.glob(str)
        if not synapseActivation:
            raise RuntimeError("Did not find any files on the specified location. Please provide list or globstring.")
    assert(isinstance(synapseActivation, list))
    
    chunks = chunkIt(synapseActivation, nprocs)
    
    myfun = lambda synapse_activation_files: _evoked_activity(cellParamName, evokedUpParamName, \
                                         synapse_activation_files, \
                                         dirPrefix = dirPrefix,
                                         tStop = tStop, scale_apical = scale_apical, post_hook = post_hook, \
                                         auto_organize_results_folder = auto_organize_results_folder)
    if silent:
        myfun = silence_stdout(myfun)
        
    d = [dask.delayed(myfun)(paths) for paths in chunks]
    return dask.delayed(lambda *args: args)(d) #return single delayed object, that computes everything



    