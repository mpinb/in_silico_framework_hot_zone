from ..context import *
from .. import decorators
from model_data_base.IO.roberts_formats import *
import model_data_base
import unittest
import dask
import tempfile
import shutil
import pandas
from pandas.util.testing import assert_frame_equal
import os



class Tests(unittest.TestCase):       
    def setUp(self):
        self.mdb = ModelDataBase(test_mdb_folder) 
        assert('synapse_activation' in self.mdb.keys())
        assert('spike_times' in self.mdb.keys())
    
    @decorators.testlevel(2)    
    def test_saved_and_reloded_synapse_file_is_identical(self):
        synapse_pdf = self.mdb['synapse_activation'].loc[self.mdb['sim_trail_index'][0]].compute(get=dask.threaded.get)
        try:
            path = tempfile.mkdtemp()
            path_file = os.path.join(path, 'test.csv')
            write_pandas_synapse_activation_to_roberts_format(path_file, synapse_pdf)
            synapse_pdf_reloaded = read_pandas_synapse_activation_from_roberts_format(path_file, sim_trail_index = self.mdb['sim_trail_index'][0]) 
        except:
            raise
        finally:
            shutil.rmtree(path)
            
        assert_frame_equal(synapse_pdf.dropna(axis=1, how = 'all'), synapse_pdf_reloaded.dropna(axis=1, how = 'all'))
