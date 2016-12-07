import os
import tempfile
import shutil
import numpy as np
import pandas as pd

from model_data_base.IO.rewrite_data_in_fast_format import _max_commas
from model_data_base.IO.rewrite_data_in_fast_format import _convert_files_csv

def convertible_to_int(x):
        try:
            int(x)
            return True
        except:
            return False
        
def write_pandas_synapse_activation_to_roberts_format(path, syn_activation):
    '''save pandas df in a format, which can be understood by the simulator'''
    with open(path, 'w') as outputFile:
        header = '# synapse type\t'
        header += 'synapse ID\t'
        header += 'soma distance\t'
        header += 'section ID\t'
        header += 'section pt ID\t'
        header += 'dendrite label\t'
        header += 'activation times\n' 
        outputFile.write(header)
        
        columns_containing_activation_times = [c for c in syn_activation.columns if convertible_to_int(c)]
        for index, row in syn_activation.iterrows():
            line = str(row['synapse_type'])
            line += '\t'
            line += str(row['synapse_ID'])
            line += '\t'
            line += str(row['soma_distance'])
            line += '\t'
            line += str(row['section_ID'])
            line += '\t'
            line += str(row['section_pt_ID'])
            line += '\t'
            line += str(row['dendrite_label'])
            line += '\t'
            activation_times = row[columns_containing_activation_times]
            for c in columns_containing_activation_times:
                t = row[c]
                if np.isnan(t):
                    break                
                line += str(t)
                line += ','

            line += '\n'
            outputFile.write(line)            

def read_pandas_synapse_activation_from_roberts_format(path, sim_trail_index = 'no_sim_trail_assigned'):
    '''reads synapse activation file from simulation and converts it to pandas table'''
    max_commas = _max_commas(path, lambda x: x)
    header = 'sim_trail_index,synapse_type,synapse_ID,soma_distance,section_ID,section_pt_ID,dendrite_label,'    
    prefix1 = os.path.dirname(path)
    prefix2 = tempfile.mkdtemp()
    fname = os.path.basename(path)
    _convert_files_csv(prefix1, prefix2, '', sim_trail_index, header, fname, max_commas)
    df = pd.read_csv(os.path.join(prefix2, fname))
    shutil.rmtree(prefix2)
    return df