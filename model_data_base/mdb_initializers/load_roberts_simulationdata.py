'''
the initialize function in this module is meant to be used to load in simulation results,
that are in the same format as roberts L6 simulations'''

from .. import IO
from .. import analyze



def _build_db_part1(mdb):
    '''builds the metadata object and rewrites files for fast access.
    Only needs to be called once to put the necessary files in the tempdir'''
    print('building database ...')
    #make filelist of all soma-voltagetraces-files
    mdb['file_list'] = IO.make_file_list(mdb['simresult_path'], 'vm_all_traces.csv')
    print('done with filelist ...')        
    #read all soma voltage traces in dask dataframe
    mdb['voltage_traces'] = IO.read_voltage_traces(mdb['simresult_path'], mdb['file_list'])
    print('done with voltage_traces ...')        
    #the indexes of this dataframe are stored for further use to identify the 
    #simulation trail
    mdb['sim_trails'] = mdb['voltage_traces'].index.compute()
    print('unambiguous sim_trail_indices generated ...')        
    #builds the metadata object, which connects the sim_trail indexes with the 
    #associated files
    
    
    ##todo: maybe the following is initializer specific and should be put in here 
    ##rather than in the general Model Data Base IO?
    mdb['metadata'] = IO.create_metadata(mdb['sim_trails']) 
    
    print('finished generating metadata ...')        
    #rewrites the synapse and cell files in a way they can be acessed fast
    print('start rewriting synapse and cell activation data in optimized format')                
    IO.rewrite_data_in_fast_format(mdb)    
    print('data is written. The above steps will not be necessary again if the' \
          + 'ModelDataBase object is instantiated in the same way.')                
    

def _build_db_part2(mdb):
    mdb['spike_times'] = analyze.spike_detection(mdb['voltage_traces'])
    mdb['synapse_activation'] = IO.read_synapse_activation_times(mdb) ##todo: maybe instead of mdb, filelist should be passed?
    mdb['cell_activation'] = IO.read_cell_activation_times(mdb)
    
    
def _regenerate_data(mdb):
    mdb['voltage_traces'] = IO.read_voltage_traces(mdb['simresult_path'], mdb['file_list'])
    _build_db_part2(mdb)  
    
def init(mdb, simresult_path):
    mdb['simresult_path'] = simresult_path  
    _build_db_part1(mdb)
    _build_db_part2(mdb)
    
    