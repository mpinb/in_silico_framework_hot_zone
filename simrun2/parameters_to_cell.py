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
import single_cell_analyzer as sca
import numpy as np
h = neuron.h
import dask
from .seed_manager import get_seed
from .utils import *

def parameters_to_cell(neuronParameters, evokedUpNWParameters, synfile, simName = '', \
                     dirPrefix = '', tStop = 345.0, scale_apical = scale_apical, post_hook = {}, allPoints=False):
    
    cellParam = neuronParameters.neuron
    paramEvokedUp = evokedUpNWParameters.network

    cell = scp.create_cell(cellParam, scaleFunc=scale_apical, allPoints=allPoints)
    
    neuronParameters.sim.tStop = tStop
    dt = neuronParameters.sim.dt

    synParametersEvoked = paramEvokedUp

    evokedNW = scp.NetworkMapper(cell, synParametersEvoked, neuronParameters.sim)
    evokedNW.reconnect_saved_synapses(synfile)

    synTypes = cell.synapses.keys()
    synTypes.sort()

    tVec = h.Vector()
    tVec.record(h._ref_t)
    cell.t = tVec
    scp.init_neuron_run(neuronParameters.sim, vardt=False) #trigger the actual simulation
    return cell