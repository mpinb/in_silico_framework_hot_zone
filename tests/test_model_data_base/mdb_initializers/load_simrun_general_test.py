import unittest
import tempfile
import warnings
from ..context import fresh_mdb
from .. import decorators
from model_data_base.mdb_initializers.load_simrun_general import optimize
from model_data_base.IO.LoaderDumper import dask_to_csv, dask_to_msgpack, dask_to_categorized_msgpack
from model_data_base.utils import silence_stdout
import distributed

optimize = silence_stdout(optimize) 
        
def test_optimization_works_dumpers_default(self, fresh_mdb):
    optimize(fresh_mdb, dumper = None, client = distributed.client_object_duck_typed)
    
def test_optimization_works_dumpers_csv(self, fresh_mdb):
    optimize(fresh_mdb, dumper = dask_to_csv, client = distributed.client_object_duck_typed)
        
def test_optimization_works_dumpers_msgpack(self, fresh_mdb):
    optimize(fresh_mdb, dumper = dask_to_msgpack, client = distributed.client_object_duck_typed)     
        
def test_optimization_works_dumpers_categorized_msgpack(self, fresh_mdb):
    optimize(fresh_mdb, dumper = dask_to_categorized_msgpack, client = distributed.client_object_duck_typed)                
        
@decorators.testlevel(2)            
def test_dataintegrity_no_empty_rows(self, fresh_mdb):
    e = fresh_mdb
    synapse_activation = e['synapse_activation']
    cell_activation = e['cell_activation']
    voltage_traces = e['voltage_traces']
    with warnings.catch_warnings():
        synapse_activation['isnan']=synapse_activation.soma_distance.apply(lambda x: np.isnan(x))
        cell_activation['isnan']=cell_activation['0'].apply(lambda x: np.isnan(x)) 
        first_column = e['voltage_traces'].columns[0]
        voltage_traces['isnan']=voltage_traces[first_column].apply(lambda x: np.isnan(x)) 
        
        assert 0 == len(synapse_activation[synapse_activation.isnan == True])
        assert 0 == len(cell_activation[cell_activation.isnan == True])
        assert 0 == len(voltage_traces[voltage_traces.isnan == True])

@decorators.testlevel(2)       
def test_voltage_traces_have_float_indices(self, fresh_mdb):
    e = fresh_mdb
    assert isinstance(e['voltage_traces'].columns[0], float)
    assert isinstance(e['voltage_traces'].head().columns[0], float)       
    
@decorators.testlevel(2)       
def test_every_entry_in_initialized_mdb_can_be_serialized(self, fresh_mdb):
    import cloudpickle
    e = fresh_mdb
    for k in e.keys():
        v = e[k]
        cloudpickle.dumps(v) # would raise an error if not picklable