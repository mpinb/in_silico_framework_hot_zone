import pandas as pd
import os


def voltage_trace_file_list(x):
    path = x.sim_trail_index
    #print path
    path, trailnr = os.path.split(path)
    path, voltage_traces_file_name = os.path.split(path)
    return pd.Series({'path': path, 'trailnr': trailnr, 'voltage_traces_file_name': voltage_traces_file_name})


def synaptic_file_list(x):
    '''return file list containig the synapse activation files belongig to
    the timetraces'''
    synapses_file_name = "simulation_run%04d_synapses.csv" % int(x.trailnr)
    cell_file_name = "simulation_run%04d_presynaptic_cells.csv" % int(x.trailnr)
    return pd.Series({'synapses_file_name': synapses_file_name, 'cell_file_name': cell_file_name})

def create_metadata(sim_trail_index):
    sim_trail_index_pd=pd.DataFrame({'sim_trail_index': list(sim_trail_index)})    
    path_trailnr = sim_trail_index_pd.apply(voltage_trace_file_list, axis = 1)
    #meta = {'synapses_file_name': 'string', 'cell_file_name': 'string'}   
    synaptic_files = path_trailnr.apply(synaptic_file_list, axis = 1)     
    sim_trail_index_complete = pd.concat((sim_trail_index_pd, path_trailnr, synaptic_files), axis = 1)
    return sim_trail_index_complete