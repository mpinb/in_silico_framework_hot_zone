import os
import tempfile
import shutil
import numpy as np
import pandas as pd
from ..utils import convertible_to_int

        
def _max_commas(path):
    '''for optimal performance, every single file should contain the same
    numer of columns. Therefore, before the data can be rewritten in
    an optimized form as csv, the maximum number of rows in all simulation
    trails in the project has to be determined.'''
    with open(path, 'r') as f:
        text = f.read()
        text = text.replace('\t',',') #only , should be used
        commas_linewise = []
        for l in text.split('\n'):
            commas_linewise.append(l.count(','))
        max_commas = max(commas_linewise)
    return max_commas

def _convert_files_csv(prefix, prefix2, path, sim_trail, header, fname, max_commas):
    #make directories
    if not os.path.exists(os.path.join(prefix2, path)):
        os.makedirs(os.path.join(prefix2, path))
    #read file in and convert it
    with open(os.path.join(prefix, path, fname), 'r') as synFile:
        text = synFile.read()
    #remove leading or trailing whitespace
    text = text.strip()
    #only use , as seperator
    text = text.replace('\t',',') #only , should be used

    max_commas = max_commas + 1 #+1 because of one additional field (sim_trail)
    #every line needs to have the same number of fields
    text_with_commas = []
    for lv, l in enumerate(text.split('\n')):
        if lv == 0: #header
            if not header[-1] == ',': header = header + ','                
            for x in range(max_commas - header.count(',') + 1):
                header = header + str(x) + ','
            text_with_commas.append(header[:-1]) #remove last comma
        else: #data
            text_with_commas.append(sim_trail + ',' + l) 
    text = '\n'.join(text_with_commas)
    #write new file
    with open(os.path.join(prefix2, path, fname), 'w+') as synFile:
        synFile.write(text)
        #print os.path.join(prefix2, path, fname)
    #print(os.path.join(prefix2, path, fname))
    return 1  
        
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

def read_pandas_synapse_activation_from_roberts_format(path, sim_trail_index = 'no_sim_trail_assigned', max_commas = None):
    '''reads synapse activation file from simulation and converts it to pandas table'''
    
    if max_commas is None:
        max_commas = _max_commas(path)#, lambda x: x)
    else:
        pass
    header = 'sim_trail_index,synapse_type,synapse_ID,soma_distance,section_ID,section_pt_ID,dendrite_label,'    
    prefix1 = os.path.dirname(path)
    prefix2 = tempfile.mkdtemp()
    fname = os.path.basename(path)
    _convert_files_csv(prefix1, prefix2, '', sim_trail_index, header, fname, max_commas)
    df = pd.read_csv(os.path.join(prefix2, fname), index_col = 'sim_trail_index')
    shutil.rmtree(prefix2)
    return df

from ..utils import split_file_to_buffers, first_line_to_key
def read_InputMapper_summary(pathOrBuffer, sep = '\t'):
    '''Expects the path to a summary csv file of the Single Cell Mapper,
    returns the tables as pandas tables'''
    def fun(f):
        '''helper function, contains the main functionality, but only accepts Buffer'''
        tables = split_file_to_buffers(f)
        tables = first_line_to_key(tables)
        for name in tables:
            tables[name] = pd.read_csv(tables[name], sep = sep)
        return tables
    
    try: #assume it is path
        with open(pathOrBuffer) as f:
            return fun(f)
    except TypeError: #if it was buffer instead
        return fun(pathOrBuffer)


