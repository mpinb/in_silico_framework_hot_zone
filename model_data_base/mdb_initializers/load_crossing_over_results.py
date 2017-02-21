from dask.diagnostics import ProgressBar
import  load_roberts_simulationdata
import model_data_base
from model_data_base.model_data_base import get_progress_bar_function


def pipeline(mdb, spikes = True, bursts = False):
    '''access to spike times and bursts'''
    with get_progress_bar_function()(): 
        if spikes:
            from ..analyze.spike_detection import spike_detection
            mdb['spike_times'] = spike_detection(mdb['voltage_traces'])
        if bursts: 
            load_roberts_simulationdata.load_dendritic_voltage_traces(mdb, dumper = 'self')    
            from ..analyze.burst_detection import burst_detection
            burst_detection(mdb['Vm_proximal'], mdb['spike_times'], burst_cutoff = -55)
        
def init_minimal(mdb, simresult_path):
    '''only access to voltage traces and spikes. Automatically
    repartitions the voltagetraces'''
    with get_progress_bar_function()(): 
        mdb['simresult_path'] = simresult_path  
        load_roberts_simulationdata._build_db_part1(mdb, repartition = True)
        from ..analyze.spike_detection import spike_detection
        mdb['spike_times'] = spike_detection(mdb['voltage_traces'])        
        print('Initialization succesful.') 
        
def init_complete(mdb, simresult_path):
    '''access to voltage traces'''
    with get_progress_bar_function()(): 
        mdb['simresult_path'] = simresult_path  
        load_roberts_simulationdata._build_db_part1(mdb, repartition = False)
        load_roberts_simulationdata._build_db_part2(mdb)
        load_roberts_simulationdata._build_db_part3(mdb)
        load_roberts_simulationdata._tidy_up(mdb)
        print('Initialization succesful.') 