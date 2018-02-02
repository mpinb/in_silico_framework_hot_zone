'''
Created on Jan 28, 2013

ongoing activity L2 neuron model

@author: robert, arco
'''
from _matplotlib_import import *
import sys
import time
import os, os.path
import glob
#for some reason, the neuron import works on the florida servers only works if tables was imported first
import tables 
import neuron
import single_cell_parser as scp
import single_cell_analyzer as sca
import numpy as np
h = neuron.h
import dask
from silence_stdout import silence_stdout
from seed_manager import get_seed

def scale_apical(cell):
    '''
    scale apical diameters depending on
    distance to soma; therefore only possible
    after creating complete cell
    '''
    dendScale = 2.5
    scaleCount = 0
    for sec in cell.sections:
        if sec.label == 'ApicalDendrite':
            dist = cell.distance_to_soma(sec, 1.0)
            if dist > 1000.0:
                continue
#            for cell 86:
            if scaleCount > 32:
                break
            scaleCount += 1
#            dummy = h.pt3dclear(sec=sec)
            for i in range(sec.nrOfPts):
                oldDiam = sec.diamList[i]
                newDiam = dendScale*oldDiam
                h.pt3dchange(i, newDiam, sec=sec)
#                x, y, z = sec.pts[i]
#                sec.diamList[i] = sec.diamList[i]*dendScale
#                d = sec.diamList[i]
#                dummy = h.pt3dadd(x, y, z, d, sec=sec)
    
    print 'Scaled %d apical sections...' % scaleCount
    
    
def _evoked_activity(cellParamName, evokedUpParamName, simName = '', dirPrefix = '', seed = None, nSweeps = 1000, tStop = 345.0, \
                    tStim = 245.0, scale_apical = scale_apical):
    '''
    pre-stimulus ongoing activity
    and evoked activity
    
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
    assert seed is not None
    np.random.seed(seed)
    print("seed: %i" % seed)
    import neuron
    neuron.load_mechanisms('/nas1/Data_arco/L5tt_blubb/mechanisms/')
    neuron.load_mechanisms('/nas1/Data_arco/project_src/simrun/simrun/mechanisms/netcon')


    neuronParameters = scp.build_parameters(cellParamName)
    evokedUpNWParameters = scp.build_parameters(evokedUpParamName) ##sumatra function for reading in parameter file
    scp.load_NMODL_parameters(neuronParameters)
    scp.load_NMODL_parameters(evokedUpNWParameters)
    cellParam = neuronParameters.neuron
    paramEvokedUp = evokedUpNWParameters.network

    cell = scp.create_cell(cellParam, scaleFunc=scale_apical)
            
    uniqueID = 'seed' + str(seed)#os.getpid()
    dirName = os.path.join(dirPrefix, 'results', \
                           time.strftime('%Y%m%d-%H%M') + '_' + str(uniqueID))
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    
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
    while nRun < nSweeps:
        synParametersEvoked = paramEvokedUp
        
        startTime = time.time()
        evokedNW = scp.NetworkMapper(cell, synParametersEvoked, neuronParameters.sim)
        evokedNW.create_saved_network2()
        stopTime = time.time()
        setupdt = stopTime - startTime
        print 'Network setup time: %.2f s' % setupdt
                
        synTypes = cell.synapses.keys()
        synTypes.sort()
        
        print 'Testing evoked response properties run %d of %d' % (nRun+1, nSweeps)
        tVec = h.Vector()
        tVec.record(h._ref_t)
        startTime = time.time()
        scp.init_neuron_run(neuronParameters.sim, vardt=False) #trigger the actual simulation
        stopTime = time.time()
        simdt = stopTime - startTime
        print 'NEURON runtime: %.2f s' % simdt
        
        vmSoma = np.array(cell.soma.recVList[0])
        t = np.array(tVec)
        vTraces.append(np.array(vmSoma[offsetBin:])), tTraces.append(np.array(t[offsetBin:]))
        for RSManager in recSiteManagers:
            RSManager.update_recordings()
        
        print 'writing simulation results'
        fname = 'simulation'
        fname += '_run%04d' % nRun
        
        synName = dirName + '/' + fname + '_synapses.csv'
        print 'computing active synapse properties'
        sca.compute_synapse_distances_times(synName, cell, t, synTypes) #calls scp.write_synapse_activation_file
        preSynCellsName = dirName + '/' + fname + '_presynaptic_cells.csv'
        scp.write_presynaptic_spike_times(preSynCellsName, evokedNW.cells)
        
        nRun += 1
        
        cell.re_init_cell()
        evokedNW.re_init_network()

        print '-------------------------------'
    
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
    
    print 'writing simulation parameter files'
    neuronParameters.save(os.path.join(dirName, uniqueID + '_neuron_model.param'))
    evokedUpNWParameters.save(os.path.join(dirName, uniqueID+ '_network_model.param'))
    return dirName

class defaultValues:
    name = 'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center'
    cellParamName = '/nas1/Data_regger/AXON_SAGA/Axon4/PassiveTouch/L5tt/network_embedding/postsynaptic_location/3x3_C2_sampling/C2center/86_CDK_20041214_BAC_run5_soma_Hay2013_C2center_apic_rec.param'
    networkName = 'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_active_ex_timing_C2center.param'
    
def run_new_simulations(cellParamName, evokedUpParamName, simName = '', dirPrefix = '', \
                                 nSweeps = 1000, nprocs = 40, tStop = 345, silent = True, \
                                 scale_apical = scale_apical):
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
        nSweeps: number of synapse activations per process
        nprocs: number of independent processes
        tStop: time in ms at which the synaptic input should stop.
        
    Returns: Delayed object. Can be computed with arbitrary scheduler.'''
    
    myfun = lambda seed: _evoked_activity(cellParamName, evokedUpParamName, simName = simName, \
                                         dirPrefix = dirPrefix, seed = seed, nSweeps = nSweeps, \
                                         tStop = tStop, scale_apical = scale_apical)
    if silent:
        myfun = silence_stdout(myfun)
        
    d = [dask.delayed(myfun)(get_seed()) for i in range(nprocs)]
    return dask.delayed(lambda *args: args)(d) #return single delayed object, that computes everything    