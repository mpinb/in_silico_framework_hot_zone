import unittest
import tempfile
import warnings
from ..context import *
from .. import decorators
from model_data_base.mdb_initializers.load_simrun_general \
            import optimize as optimize_simrun_general
from model_data_base.mdb_initializers.synapse_activation_binning \
            import init as init_synapse_activation

from model_data_base.IO.LoaderDumper import dask_to_csv, dask_to_msgpack, dask_to_categorized_msgpack
from model_data_base.utils import silence_stdout

optimize_simrun_general = silence_stdout(optimize_simrun_general)

class Tests(unittest.TestCase):
    def setUp(self):
        # set up model_data_base in temporary folder and initialize it.
        # This additionally is an implicit test, which ensures that the
        # initialization routine does not throw an error
        self.fmdb = FreshlyInitializedMdb()
        self.mdb = self.fmdb.__enter__()
        optimize_simrun_general(self.mdb)
                       
    def tearDown(self):
        self.fmdb.__exit__()
        
    def test_API(self):
        init_synapse_activation(self.mdb, groupby = 'EI')
        init_synapse_activation(self.mdb, groupby = ['EI'])
        init_synapse_activation(self.mdb, groupby = ['EI', 'proximal'])        
        
        