import os, os.path
#os.chdir('/nas1/Data_arco/simrun/simrun/')

import matplotlib
matplotlib.use('Agg')
import sys
import time

import glob, shutil
import neuron
import single_cell_parser as scp
import single_cell_analyzer as sca
from model_data_base.IO.roberts_formats import write_pandas_synapse_activation_to_roberts_format
import numpy as np
import pandas as pd
h = neuron.h
#neuron.load_mechanisms('/nas1/Data_regger/AXON_SAGA/Axon4/PassiveTouch/L5tt/first_testing/mechanisms')
neuron.load_mechanisms('/nas1/Data_arco/L5tt_blubb/mechanisms/')
neuron.load_mechanisms('/nas1/Data_arco/project_src/simrun/simrun/mechanisms/netcon')
#neuron.load_mechanisms('/home/regger/bin/nrn-7.2/share/examples/nrniv/netcon')


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

def simtrail_to_cell_object(mdb, sim_trail_index, compute = True, allPoints = False, \
                            scale_apical = scale_apical, range_vars = None, silent = True):
    '''Resimulates simulation trail and returns cell object.
    Expects Instance of ModelDataBase and sim_trail index.
    The mdb has to contain the paths to the parameterfiles at the following location: 
        ('parameterfiles', 'cellName')
        ('parameterfiles', 'networkName')
        
    Keyword arguments:
        compute (default: True): If true, a cell object is returned. If false, the respective
                            delayed object is returned
        allPoints (default: False): if True, the cell will be simulated at very high resolution.
                            This mode might be useful for high quality visualizations.
        scale_apical: function that gets called after the cell is set up. 
        range_vars: range variables that should be recorded
        silent: If True, nothing will be written to stdout                    
    '''
    stdout_bak = sys.stdout
    if silent == True:
        sys.stdout = open(os.devnull, "w")
        
    try:    
        metadata = mdb['metadata']
        metadata = metadata[metadata.sim_trail_index == sim_trail_index]
        assert(len(metadata) == 1)
        m = metadata.iloc[0]
        synapse_activation_file = os.path.join(mdb['simresult_path'], m['path'], m['synapses_file_name'])
        dummy =  trail_to_cell_object(cellName = mdb[('parameterfiles', 'cellName')], \
                                    networkName = mdb[('parameterfiles', 'networkName')], \
                                    synapse_activation_file = synapse_activation_file, \
                                    range_vars = range_vars, 
                                    scale_apical = scale_apical, 
                                    allPoints = allPoints, 
                                    compute = compute)
    finally:
        if silent == True:
            sys.stdout = stdout_bak

    return dummy

import tempfile
def trail_to_cell_object(name = None, cellName = None, networkName = None, synapse_activation_file = None, \
                    range_vars = None, scale_apical = scale_apical, allPoints = False, compute = True):
    tempdir = None

    try:
        #if pandas dataframe instead of path is given: convert pandas dataframe to file
        if isinstance(synapse_activation_file, pd.DataFrame):
            tempdir = tempfile.mkdtemp()
            
            write_pandas_synapse_activation_to_roberts_format(os.path.join(tempdir, 'synfile.csv'), \
                                                              synapse_activation_file)
            
            synfile = os.path.join(tempdir, 'synfile.csv')
        elif isinstance(synapse_activation_file, str):
            synfile = synapse_activation_file
        #set up simulation            
        simName = name
        cellName = cellName
        evokedUpParamName = networkName
        
        neuronParameters = scp.build_parameters(cellName)
        evokedUpNWParameters = scp.build_parameters(evokedUpParamName) ##sumatra function for reading in parameter file
        scp.load_NMODL_parameters(neuronParameters)
        scp.load_NMODL_parameters(evokedUpNWParameters)
        cellParam = neuronParameters.neuron
        paramEvokedUp = evokedUpNWParameters.network
        vTraces = []
        tTraces = []   
        
        #    cell = scp.create_cell(cellParam, scaleFunc=scale_apical)
        cell = scp.create_cell(cellParam, scaleFunc=scale_apical, allPoints=allPoints)
        cell.re_init_cell()
        
        
        tOffset = 0.0 # avoid numerical transients
        tStim = 245.0
        tStop = 345.0
        neuronParameters.sim.tStop = tStop
        dt = neuronParameters.sim.dt
        offsetBin = int(tOffset/dt + 0.5)
        nRun = 0
        synParametersEvoked = paramEvokedUp
        startTime = time.time()
        evokedNW = scp.NetworkMapper(cell, synParametersEvoked, neuronParameters.sim)
        evokedNW.re_init_network()    
        evokedNW.reconnect_saved_synapses(synfile)
        
        if range_vars is not None:
            if isinstance(range_vars, str):
                range_vars = [range_vars]
            for var in range_vars:
                cell.record_range_var(var)
        
        if compute:
            tVec = h.Vector()
            tVec.record(h._ref_t)
            startTime = time.time()
            scp.init_neuron_run(neuronParameters.sim, vardt=False) #trigger the actual simulation
            stopTime = time.time()
            simdt = stopTime - startTime
            print 'NEURON runtime: %.2f s' % simdt
            t = np.array(tVec)
            vmSoma = np.array(cell.soma.recVList[0])
            cell.t = np.array(t[offsetBin:])
            cell.vmSoma = np.array(vmSoma[offsetBin:])
    except:
        raise
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir)                      
    return cell

