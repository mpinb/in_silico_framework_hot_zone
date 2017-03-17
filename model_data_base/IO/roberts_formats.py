import os
#from six import BytesIO
from StringIO import StringIO as BytesIO
import numpy as np
import pandas as pd
from model_data_base.utils import convertible_to_int, split_file_to_buffers, first_line_to_key



#################################################################
# read synapse and cell activation file
#################################################################
def _process_line_fun(line, n_commas):
    line = line.strip()
    line = line.replace('\t', ',')
    real_n_commas = line.count(',')
    return line + ',' * (n_commas-real_n_commas)+'\n'

def _replace_commas(f, n_commas, header, skiprows = 1):
    f2 = BytesIO()
    f.seek(0)
    header = header + ','.join([str(x) for x in range(n_commas-header.count(','))])+'\n'
    f2.write(header)
    for line in f.read().split('\n')[skiprows:]:
        line = _process_line_fun(line, n_commas)
        f2.write(line)
    f2.seek(0)
    return f2

def read_csv_uneven_length(path, n_commas, header = None, skiprows = 0):
    with open(path) as f:
        bla = _replace_commas(f, n_commas, header, skiprows)
    return pd.read_csv(bla, index_col = False)

def _max_commas(path):
    with open(path, 'r') as f:
        text = f.read()
        text = text.replace('\t',',') #only , should be used
        commas_linewise = []
        for l in text.split('\n'):
            if not l:
                continue
            comma_at_end = l[-1] == ',' # allways assume that line ends with comma
            commas_linewise.append(l.count(',') + int(not comma_at_end))
        max_commas = max(commas_linewise)
    return max_commas

def _read_roberts_csv_uneven_length_helper(path, header, sim_trail_index = 'no_sim_trail_assigned', \
                                           max_commas = None, set_index = True):
    '''general function for reading roberts csv files that have a variable amount of delimiters. 
    Also supports vectorized arguments for path and sim_trail_index.
    '''
    if isinstance(path, (list, tuple)): 
        #checks for the vectorized case
        assert(isinstance(sim_trail_index, (list, tuple)))
        assert(max_commas is not None)
    else:
        path = [path]
        sim_trail_index = [sim_trail_index]

    if max_commas is None: max_commas = _max_commas(path)
    def fun(path, sim_trail_index):
        '''read single file'''        
        df = read_csv_uneven_length(path, max_commas, header = header, skiprows = 1)
        df['sim_trail_index'] = sim_trail_index
        return df
     
    p_sti_tuples = zip(path, sim_trail_index)

    df = pd.concat([fun(p, sti) for p, sti in p_sti_tuples])
    if set_index: df.set_index('sim_trail_index', inplace = True)
    return df

def read_pandas_synapse_activation_from_roberts_format(path, sim_trail_index = 'no_sim_trail_assigned', 
                                                       max_commas = None, set_index = True):
    '''reads synapse activation file from simulation and converts it to pandas table'''
    header = 'synapse_type,synapse_ID,soma_distance,section_ID,section_pt_ID,dendrite_label,'
    return _read_roberts_csv_uneven_length_helper(path, header, sim_trail_index, max_commas, set_index)

def read_pandas_cell_activation_from_roberts_format(path, sim_trail_index = 'no_sim_trail_assigned', \
                                                    max_commas = None, set_index = True):
    '''reads cell activation file from simulation and converts it to pandas table'''
    header = 'sim_trail_index,presynaptic_cell_type,cell_ID'
    return _read_roberts_csv_uneven_length_helper(path, header, sim_trail_index, max_commas, set_index)

############################################################
# write synapse activation file
############################################################
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


############################################################
# read input mapper summary
############################################################
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


