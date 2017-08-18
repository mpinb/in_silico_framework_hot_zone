from ..context import *
from .. import decorators
from model_data_base.mdb_initializers.load_simrun_general import *
import unittest
import os, shutil
from mock import MagicMock
import numpy as np
import tempfile
import model_data_base

class Tests(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.mdb = model_data_base.ModelDataBase(self.path)
        self.mdb.settings.show_computation_progress = False        
    def tearDown(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    @decorators.testlevel(2)        
    def test_can_build_database(self):
        mdb = self.mdb
        init(mdb, test_data_folder)
        optimize(mdb)
        assert(isinstance(mdb['spike_times'].iloc[0][0], float))
        assert(isinstance(mdb['voltage_traces'].head().iloc[0][0], float))
        self.assertIsInstance(mdb['voltage_traces'].columns[0], float)
        self.assertIsInstance(mdb['voltage_traces'].head().columns[0], float)