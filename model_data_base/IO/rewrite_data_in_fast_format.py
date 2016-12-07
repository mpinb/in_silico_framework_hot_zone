import os
import pandas as pd
import dask
import dask.dataframe as dd
from .. import settings

def _max_commas(x, pathgenerator):
    '''for optimal performance, every single file should contain the same
    numer of columns. Therefore, before the data can be rewritten in
    an optimized form as csv, the maximum number of rows in all simulation
    trails in the project has to be determined.'''
    path = pathgenerator(x)
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
    #print(max_commas)
    #print(max_commas)
    #print(type(max_commas))

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
            #print(sim_trail + ',' + l)
            #text_with_commas.append(l + ','*(max_commas - l.count(',') - 10))
    text = '\n'.join(text_with_commas)
    #write new file
    with open(os.path.join(prefix2, path, fname), 'w+') as synFile:
        synFile.write(text)
        #print os.path.join(prefix2, path, fname)
    #print(os.path.join(prefix2, path, fname))
    return 1  
    
class ConverterFabric:
    def __init__(self, mdb):
        self.mdb = mdb
    
    def set_filetype(self,filetype):
        self.filetype = filetype
    
    def get_pathgenerator(self):
        if self.filetype == 'cells':
            def pathgenerator(x):
                path = os.path.join(self.mdb.path, x.path, x.cell_file_name)
                return path
            return pathgenerator
        if self.filetype == 'synapses':
            def pathgenerator(x):
                path = os.path.join(self.mdb.path, x.path, x.synapses_file_name)
                return path
            return pathgenerator            
    
    def get_max_commas_fun(self):
        return lambda x: _max_commas(x, self.get_pathgenerator())
    
    def get_convert_fun(self, max_commas):
        if self.filetype == 'cells':
            def convert_fun(x):
                path = x.path
                fname = x.cell_file_name
                sim_trail = x.sim_trail_index
                _convert_files_csv(self.mdb.path, self.mdb.tempdir, path, sim_trail, 'sim_trail_index,presynaptic_cell_type,cell_ID,', fname, max_commas)
                return 1 #return something, so dask will not complain
            
            return convert_fun
        if self.filetype == 'synapses':
            def convert_fun(x):
                path = x.path
                fname = x.synapses_file_name
                sim_trail = x.sim_trail_index
                _convert_files_csv(self.mdb.path, self.mdb.tempdir, path, sim_trail, 'sim_trail_index,synapse_type,synapse_ID,soma_distance,section_ID,section_pt_ID,dendrite_label,', fname, max_commas)
                return 1 #return something, so dask will not complain
            return convert_fun

def rewrite_data_in_fast_format(mdb):
    scheduler = settings.multiprocessing_scheduler
    metadata_dd = dd.from_pandas(mdb.metadata, npartitions = settings.npartitions)
    myConvFab = ConverterFabric(mdb)
    
    myConvFab.set_filetype('cells')
    max_commas = metadata_dd.apply(myConvFab.get_max_commas_fun(), 
                                             axis = 1, 
                                             meta = int).compute(get = scheduler)#, meta = pd.Series({'max_commas': 'int64'}))
    max_commas = max(list(max_commas))
    metadata_dd.apply(myConvFab.get_convert_fun(max_commas), 
                      axis = 1, meta = 1).compute(get = scheduler)
    
    
    myConvFab.set_filetype('synapses')
    max_commas = metadata_dd.apply(myConvFab.get_max_commas_fun(), 
                                   axis = 1, 
                                   meta = 1).compute(get = scheduler)
    max_commas = max(list(max_commas))
    metadata_dd.apply(myConvFab.get_convert_fun(max_commas), 
                      axis = 1, meta = 1).compute(get = scheduler)    
