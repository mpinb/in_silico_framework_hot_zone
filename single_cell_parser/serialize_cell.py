'''
Created 2016/2017

@author: arco
'''

import numpy as np
import pandas as pd
import sys, os

from model_data_base.utils import silence_stdout
import single_cell_parser as scp

def convert_hoc_array_to_np_array(hoc_array):
    '''converts hoc array to list of lists'''
    return np.array(hoc_array)

def convert_dict_of_hoc_arrays_to_dict_of_np_arrays(hoc_array_dict):
    '''converts dictionary of hoc arrays to dictionary of list of lists'''
    out = {}
    for name in hoc_array_dict:
        out[name]=convert_hoc_array_to_np_array(hoc_array_dict[name])
    return out

def cell_to_serializable_object(cell, paramFile):
    '''takes cell object and returns an object, that contains the saved information
    (recorded range vars, voltage) and that can be serialized.'''
    out = []
    for lv, sec in enumerate(cell.sections):
        dummy = {}
        dummy['recVList'] = convert_hoc_array_to_np_array(sec.recVList)
        dummy['recordVars'] = convert_dict_of_hoc_arrays_to_dict_of_np_arrays(sec.recordVars)
        out.append(dummy)
    
    try:
        t = cell.t
    except:
        t = []
        
    sc = {} #serialisable object, that contains all the information required to rebuild the cell
    sc['per_section'] = out
    sc['time'] = t
    sc['param'] = paramFile
    
    return sc# out, t


def restore_cell_from_serializable_object(sc):
    cellParam = silence_stdout(scp.build_parameters)(sc['param'])
    cell = silence_stdout(scp.create_cell)(cellParam.neuron)
    cell.t = sc['time']
    for outdict, sec in zip(sc['per_section'], cell.sections):
        for name in outdict:
            setattr(sec, name, outdict[name])
    return cell

def save_cell_to_file(path, cell, paramFile):
    sc = cell_to_serializable_object(cell, paramFile)
    pd.Series([sc]).to_msgpack(path)
    
def load_cell_from_file(path):
    pds = pd.read_msgpack(path)
    sc = pds[0]
    return restore_cell_from_serializable_object(sc)