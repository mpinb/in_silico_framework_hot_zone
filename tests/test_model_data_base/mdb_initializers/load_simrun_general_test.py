import unittest
import tempfile
import warnings
from ..context import *
from .. import decorators
from model_data_base.mdb_initializers.load_simrun_general import *
from model_data_base.IO.LoaderDumper import dask_to_csv, dask_to_msgpack, dask_to_categorized_msgpack
from model_data_base.utils import silence_stdout
import distributed

optimize = silence_stdout(optimize)
init = silence_stdout(init)
client = distributed.client_object_duck_typed

assert(client is not None)

class Tests(unittest.TestCase):
    def setUp(self):
        # set up model_data_base in temporary folder and initialize it.
        # This additionally is an implicit test, which ensures that the
        # initialization routine does not throw an error
        self.fmdb = FreshlyInitializedMdb()
        self.mdb = self.fmdb.__enter__()
                       
    def tearDown(self):
        self.fmdb.__exit__()
        
    def test_optimization_works_dumpers_default(self):
        optimize(self.mdb, dumper = None, client = client)
     
    def test_optimization_works_dumpers_csv(self):
        optimize(self.mdb, dumper = dask_to_csv, client = client)
         
    def test_optimization_works_dumpers_msgpack(self):
        optimize(self.mdb, dumper = dask_to_msgpack, client = client)     
         
    def test_optimization_works_dumpers_categorized_msgpack(self):
        optimize(self.mdb, dumper = dask_to_categorized_msgpack, client = client)                
         
    @decorators.testlevel(2)            
    def test_dataintegrity_no_empty_rows(self):
        e = self.mdb
        synapse_activation = e['synapse_activation']
        cell_activation = e['cell_activation']
        voltage_traces = e['voltage_traces']
        with warnings.catch_warnings():
            synapse_activation['isnan']=synapse_activation.soma_distance.apply(lambda x: np.isnan(x))
            cell_activation['isnan']=cell_activation['0'].apply(lambda x: np.isnan(x)) 
            first_column = e['voltage_traces'].columns[0]
            voltage_traces['isnan']=voltage_traces[first_column].apply(lambda x: np.isnan(x)) 
          
            self.assertEqual(0, len(synapse_activation[synapse_activation.isnan == True]))
            self.assertEqual(0, len(cell_activation[cell_activation.isnan == True]))
            self.assertEqual(0, len(voltage_traces[voltage_traces.isnan == True]))
    
    @decorators.testlevel(2)       
    def test_voltage_traces_have_float_indices(self):
        e = self.mdb
        self.assertIsInstance(e['voltage_traces'].columns[0], float)
        self.assertIsInstance(e['voltage_traces'].head().columns[0], float)       
        
    @decorators.testlevel(2)       
    def test_every_entry_in_initialized_mdb_can_be_serialized(self):
        import cloudpickle
        e = self.mdb
        for k in e.keys():
            v = e[k]
            cloudpickle.dumps(v) # would raise an error if not picklable