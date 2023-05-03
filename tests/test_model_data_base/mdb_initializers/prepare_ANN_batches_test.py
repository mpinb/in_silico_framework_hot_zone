import unittest
import tempfile
import warnings
from ..context import *
import numpy as np
from .. import decorators
from model_data_base.mdb_initializers.prepare_ANN_batches import spike_times_to_onehot
from model_data_base.mdb_initializers.load_simrun_general \
            import optimize as optimize_simrun_general
from model_data_base.mdb_initializers.synapse_activation_binning \
            import init as init_synapse_activation

from model_data_base.IO.LoaderDumper import dask_to_csv, dask_to_msgpack, dask_to_categorized_msgpack
from model_data_base.utils import silence_stdout
import distributed

client = distributed.client_object_duck_typed

optimize_simrun_general = silence_stdout(optimize_simrun_general)

class Tests(unittest.TestCase):
    def setUp(self):
        # set up model_data_base in temporary folder and initialize it.
        # This additionally is an implicit test, which ensures that the
        # initialization routine does not throw an error
        self.fmdb = FreshlyInitializedMdb()
        self.mdb = self.fmdb.__enter__()
        optimize_simrun_general(self.mdb, client = client)
        # example spike_times
        self.spike_times = [15.999999999999625, 23.724999999999188, 30.449999999998806, 165.79999999998614, 181.62499999997175, 298.3249999998656, 319.37499999984647]
                       
    def tearDown(self):
        self.fmdb.__exit__()
        
    def test_API(self):
        init_synapse_activation(self.mdb, groupby = 'EI')
        init_synapse_activation(self.mdb, groupby = ['EI'])
        init_synapse_activation(self.mdb, groupby = ['EI', 'proximal'])        
        
    def test_onehot_encoding(self):
        time_steps=[1, 13]  # test different time step intervals for one-hot encoding
        for time_step in time_steps:
            one_hot = spike_times_to_onehot(self.spike_times, min_time=0, max_time=505, time_step=time_step)
            assert all([a == b for a, b in zip([st//time_step for st in self.spike_times],  np.where(one_hot)[0])]), \
            "One-hot encoding failed.\nSpike times: {}\none-hot coding:{}\nLocations where spike equals True:{}".format(self.spike_times, one_hot, np.where(one_hot)[0])
        try:
            spike_times_to_onehot([-1, -200])  # should give error
            assert False  # in case it does not give an error
        except AssertionError:
            assert True  # all good