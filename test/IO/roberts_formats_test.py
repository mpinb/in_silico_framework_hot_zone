from ..context import *
from .. import decorators
from model_data_base.IO.roberts_formats import *
from model_data_base import ModelDataBase
import unittest
import dask
import tempfile
import shutil
import pandas
from pandas.util.testing import assert_frame_equal
import os

class Tests(unittest.TestCase):       
    def setUp(self):
        mdb = ModelDataBase('test/data/test_temp') 
        if not 'synapse_activation' in mdb.keys():
            import model_data_base.mdb_initializers.load_roberts_simulationdata
            model_data_base.mdb_initializers.load_roberts_simulationdata.init(mdb, 'test/data/test_data')     
        if not 'spike_times' in mdb.keys():
            import model_data_base.mdb_initializers.load_roberts_simulationdata            
            model_data_base.mdb_initializers.load_roberts_simulationdata.pipeline(mdb)    
    
    @decorators.testlevel(1)    
    def test_saved_and_reloded_synapse_file_is_identical(self):
        mdb = model_data_base.ModelDataBase(os.path.join('/nas1/Data_arco/project_src/model_data_base/', 'test/data/test_temp'))
        synapse_pdf = mdb['synapse_activation'].loc[mdb['sim_trail_index'][0]].compute(get=dask.threaded.get)
        try:
            path = tempfile.mkdtemp()
            path_file = os.path.join(path, 'test.csv')
            write_pandas_synapse_activation_to_roberts_format(path_file, synapse_pdf)
            synapse_pdf_reloaded = read_pandas_synapse_activation_from_roberts_format(path_file, sim_trail_index = mdb['sim_trail_index'][0]) 
        except:
            raise
        finally:
            shutil.rmtree(path)
            
        assert_frame_equal(synapse_pdf.dropna(axis=1, how = 'all'), synapse_pdf_reloaded.dropna(axis=1, how = 'all'))
