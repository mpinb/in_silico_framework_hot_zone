from context import *

import os, sys, glob, shutil, tempfile
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd
from pandas.util.testing import assert_frame_equal
import dask
import dask.dataframe as dd
import neuron

import Interface as I

import decorators
import unittest
import numpy as np

import simrun2.generate_synapse_activations
import simrun2.run_new_simulations
import simrun2.run_existing_synapse_activations
import simrun2.sim_trail_to_cell_object
import simrun2.crossing_over.crossing_over_simple_interface
from model_data_base.IO.roberts_formats import read_pandas_synapse_activation_from_roberts_format

import getting_started

getting_started_folder = getting_started.parent

name = 'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center'
cellParamName = os.path.join(getting_started_folder, \
                             'biophysical_constraints', \
                             '86_CDK_20041214_BAC_run5_soma_Hay2013_C2center_apic_rec.param')
networkName = os.path.join(getting_started_folder, \
                           'functional_constraints', \
                           'network.param')
example_path = os.path.join(getting_started_folder, \
                            'example_simulation_data', \
                            'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center', \
                            '20150815-1530_20240', \
                            'simulation_run0000_synapses.csv')

assert(os.path.exists(cellParamName))
assert(os.path.exists(networkName))
assert(os.path.exists(example_path))

class Tests(unittest.TestCase):       
    def setUp(self):
        pass


    @decorators.testlevel(1)    
    def test_generate_synapse_activation_returns_filelist(self):
        dirPrefix = tempfile.mkdtemp()
        try:
            dummy = simrun2.generate_synapse_activations.generate_synapse_activations(cellParamName, networkName, simName='', dirPrefix=dirPrefix, nSweeps=1, nprocs=1, tStop=345, silent=True)
            dummy = dummy.compute(get = dask.async.get_sync)
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
        print dummy
        self.assertIsInstance(dummy[0][0][0], str)        
      
    @decorators.testlevel(2)    
    def test_run_existing_synapse_activation_returns_identifier_dataframe_anf_results_folder(self):        
        dirPrefix = tempfile.mkdtemp()
        try:
            dummy = simrun2.run_existing_synapse_activations.run_existing_synapse_activations(cellParamName, networkName, [example_path], simName='', dirPrefix=dirPrefix, nprocs=1, tStop=345, silent=True)
            dummy = dummy.compute(get = dask.async.get_sync)
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
        self.assertIsInstance(dummy[0][0][0], pd.DataFrame)      
        self.assertIsInstance(dummy[0][0][1], str)  
           
    @decorators.testlevel(2)    
    def test_run_new_simulations_returns_dirname(self):
        dirPrefix = tempfile.mkdtemp()
        try:
            dummy = simrun2.run_new_simulations.run_new_simulations(cellParamName, networkName, simName='', dirPrefix=dirPrefix, nSweeps=1, nprocs=1, tStop=345, silent=True)
            dummy = dummy.compute(get = dask.async.get_sync)
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
        print dummy
        self.assertIsInstance(dummy[0][0], str)  
          
    @decorators.testlevel(2)              
    def test_position_of_morphology_does_not_matter_after_network_mapping(self):  
        try:
            dirPrefix = tempfile.mkdtemp()
            dummy = simrun2.run_existing_synapse_activations.run_existing_synapse_activations(cellParamName, networkName, [example_path], simName='', dirPrefix=dirPrefix, nprocs=1, tStop=345, silent=True)
            dummy = dummy.compute(get = dask.async.get_sync)
            cellParam = I.scp.build_parameters(cellParamName)
            cellParam.neuron.filename = os.path.join(parent, 'test_simrun2',\
                         'data', \
                         '86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2_B1border.hoc')
            print cellParam.neuron.filename
            cellParamName_other_position = os.path.join(dirPrefix, 'other_position.param')                
            cellParam.save(cellParamName_other_position)
            dummy2 = simrun2.run_existing_synapse_activations.run_existing_synapse_activations(cellParamName_other_position, networkName, [example_path], simName='', dirPrefix=dirPrefix, nprocs=1, tStop=345, silent=True)
            dummy2 = dummy2.compute(get = dask.async.get_sync)
            df1 = I.read_pandas_synapse_activation_from_roberts_format(os.path.join(dummy[0][0][1], 'simulation_run%s_synapses.csv' % dummy[0][0][0].iloc[0].number))
            df2 = I.read_pandas_synapse_activation_from_roberts_format(os.path.join(dummy2[0][0][1], 'simulation_run%s_synapses.csv' % dummy[0][0][0].iloc[0].number))
            assert_frame_equal(df1, df2)            
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
          
    @decorators.testlevel(2)               
    def test_reproduce_simulation_trail_from_roberts_model_control(self):
        dirPrefix = tempfile.mkdtemp()
        try:
            dummy = simrun2.run_existing_synapse_activations.run_existing_synapse_activations(cellParamName, networkName, [example_path], simName='', dirPrefix=dirPrefix, nprocs=1, tStop=345, silent=True)
            dummy = dummy.compute(get = dask.async.get_sync)
                
            #synapse activation
            df1 = read_pandas_synapse_activation_from_roberts_format(os.path.join(dummy[0][0][1], 'simulation_run%s_synapses.csv' % dummy[0][0][0].iloc[0].number))
            df2 = read_pandas_synapse_activation_from_roberts_format(example_path)
            df1 = df1[[c for c in df1.columns if c.isdigit()] + ['synapse_type', 'soma_distance', 'dendrite_label']]
            df2 = df2[[c for c in df1.columns if c.isdigit()] + ['synapse_type', 'soma_distance', 'dendrite_label']]
            assert_frame_equal(df1, df2)
                
            #voltage traces
            path1 = glob.glob(os.path.join(dummy[0][0][1], '*_vm_all_traces*.csv'))[0]
            path2 = glob.glob(os.path.join(os.path.dirname(example_path), '*_vm_all_traces.csv'))[0]
            pdf1 = pd.read_csv(path2, sep = '\t')[['t', 'Vm run 00']]
            pdf2 = pd.read_csv(path1, sep = '\t')
            pdf2 = pdf2[pdf2['t'] >=100].reset_index(drop = True)
            assert_frame_equal(pdf1, pdf2)
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
        self.assertIsInstance(dummy[0][0][0], pd.DataFrame)      
        self.assertIsInstance(dummy[0][0][1], str)         
               
           
   
#     def test_ongoing_frequency_in_new_monte_carlo_simulation_is_like_in_roberts_control(self):
#         try:
#             pass
#         except:
#             raise
#         finally:
#             pass
#          
#          
#     def test_different_schedulers_give_same_result():
#         pass
    @decorators.testlevel(2)    
    def test_crossing_over_trails_show_identical_response_before_crossing_over_time(self):
        dirPrefix = tempfile.mkdtemp()
        import test_model_data_base.context
        try:
            t = np.random.randint(100, high = 150)
            with test_model_data_base.context.FreshlyInitializedMdb() as mdb:
                sim_trail = list(mdb['sim_trail_index'])[0]
                pdf, res = simrun2.crossing_over.crossing_over_simple_interface.crossing_over(mdb, sim_trail, t, cellParamName, networkName, dirPrefix = dirPrefix, nSweeps=2, tStop=345)
                res = res.compute(get = dask.async.get_sync)
                df = pd.read_csv(glob.glob(os.path.join(res[0][0][0][1], '*vm_all_traces.csv'))[0], sep = '\t')
                assert_almost_equal(df[df.t<t]['Vm run 00'].values, df[df.t<t]['Vm run 01'].values)
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
          
          
          