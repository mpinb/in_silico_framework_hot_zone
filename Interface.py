'''
The purpose of this module is to glue together all libraries necessary for single cell simulations
of cells in barrel cortex. This includes:

 - Moving around hoc-morphologies
 - Compute anatomical realization of presynaptic cells and synapses
 - Activation of synapses based on experimental data
 - determining apropriate biophysical parameters
 - setting up a cluster
 - use that cluster to optimize / find suitable parameters
 - use that cluster to compute single cell responses to synaptic input
 - efficiently store the simulation results and provide an easy interface to query data

The recommendet use is to import it in a jupyter notebook in the following manner:
    import Interface as I
    
'''

import os
import sys
import tempfile
import shutil
import glob
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import warnings
import traceback
import sys
import collections

try:
    from IPython import display
except ImportError:
    pass

try:
    import seaborn as sns
except ImportError:
    pass



# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#  
#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#  
# warnings.showwarning = warn_with_traceback


# todo: the version is not specific to model_data_base,
# therefore - ideally - the _version.py file would not live in the 
# subfolder mondel data base but in the top folder.
# But then it does not work anymore ...
# Currently, git needs to be installed to be able to get
# the current version. Use versioneer.py to be able 
# to bundle package without the need of having git installed.
from model_data_base._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']

print "Current version: {version}".format(version = __version__)
print "Current pid: {pid}".format(pid = os.getpid())

from model_data_base.model_data_base import ModelDataBase
from model_data_base.analyze.burst_detection import burst_detection
from model_data_base.analyze.LDA import lda_prediction_rates as lda_prediction_rates
from model_data_base.analyze.temporal_binning import universal as temporal_binning
from model_data_base.analyze.synapse_activation_binning import synapse_activation_postprocess_dask as synapse_activation_binning_dask

from model_data_base.analyze.spike_detection import spike_detection, spike_in_interval
from model_data_base.analyze.spaciotemporal_binning import universal as spaciotemporal_binning
from model_data_base.analyze import split_synapse_activation
from model_data_base.analyze.analyze_input_mapper_result import compare_to_neuronet

from model_data_base.IO.LoaderDumper import dask_to_csv as dumper_dask_to_csv
from model_data_base.IO.LoaderDumper import numpy_to_npy as dumper_numpy_to_npy
from model_data_base.IO.LoaderDumper import pandas_to_pickle as dumper_pandas_to_pickle
from model_data_base.IO.LoaderDumper import dask_to_msgpack as dumper_dask_to_msgpack
from model_data_base.IO.LoaderDumper import dask_to_categorized_msgpack as dumper_dask_to_categorized_msgpack
from model_data_base.IO.LoaderDumper import cell as dumper_cell
from model_data_base.IO.LoaderDumper import to_pickle as dumper_to_pickle
from model_data_base.IO.LoaderDumper import to_cloudpickle as dumper_to_cloudpickle


from model_data_base.IO.roberts_formats import write_pandas_synapse_activation_to_roberts_format
from model_data_base.IO.roberts_formats import read_pandas_synapse_activation_from_roberts_format
from model_data_base.IO.roberts_formats import read_pandas_cell_activation_from_roberts_format
from model_data_base.IO.roberts_formats import read_InputMapper_summary

#from model_data_base.mdb_initializers import load_crossing_over_results as mdb_init_crossing_over
#from model_data_base.mdb_initializers import load_roberts_simulationdata as mdb_init_roberts_simulations
from model_data_base.mdb_initializers import load_simrun_general as mdb_init_simrun_general
#for compatibility
mdb_init_crossing_over = mdb_init_roberts_simulations = mdb_init_simrun_general

from model_data_base.analyze import split_synapse_activation, color_cellTypeColorMap, excitatory, inhibitory
from model_data_base.utils import silence_stdout
from model_data_base.utils import select, pandas_to_array, pooled_std
from model_data_base.utils import skit, chunkIt
from model_data_base.utils import cache

try: ##to avoid import errors in distributed system because of missing matplotlib backend
    import matplotlib
    import matplotlib.pyplot as plt
    try:
        from model_data_base.plotfunctions.average_std import average_std as average_std
        from model_data_base.plotfunctions.histogram import histogram as histogram
        from model_data_base.plotfunctions.manylines import manylines
        from model_data_base.plotfunctions.rasterplot import rasterplot
        from model_data_base.plotfunctions.cell_to_ipython_animation import cell_to_ipython_animation, display_animation
        from model_data_base.plotfunctions._figure_array_converter import show_pixel_object, PixelObject
    except ImportError:
        print "Could not import plotfunctions!"
except ImportError:
    print "Could not import matplotlib!"

try:
    from simrun2.run_existing_synapse_activations import run_existing_synapse_activations \
        as simrun_run_existing_synapse_activations
    from simrun2.generate_synapse_activations import generate_synapse_activations \
        as simrun_generate_synapse_activations
    from simrun2.run_new_simulations import run_new_simulations \
        as simrun_run_new_simulations
    from simrun2.sim_trail_to_cell_object import simtrail_to_cell_object \
        as simrun_simtrail_to_cell_object
    from simrun2.sim_trail_to_cell_object import trail_to_cell_object \
        as simrun_trail_to_cell_object
    from simrun2 import crossing_over as simrun_crossing_over_module
    from simrun2.crossing_over.crossing_over_simple_interface import crossing_over as simrun_crossing_over_simple_interface
except ImportError:
    print "Could not import full-compartmental-model stuff"

from simrun2.reduced_model import synapse_activation \
    as rm_synapse_activations
#from simrun2.reduced_model import spiking_output \
#    as simrun_reduced_model_spiking_output
from simrun2.reduced_model import get_kernel \
    as rm_get_kernel

import single_cell_analyzer as sca
import single_cell_parser as scp

from singlecell_input_mapper.map_singlecell_inputs import map_singlecell_inputs

if get_versions()['dirty']: warnings.warn('The source folder has uncommited changes!')

try:
    import distributed
    @cache
    def cluster(*args, **kwargs):
        c = distributed.Client(*args, **kwargs)
        # import matplotlib to avoid error with missing Qt backend
        def fun():
            import matplotlib
            matplotlib.use('Agg')
        c.run(fun)
        return c
    print "setting up local multiprocessing framework ... done"
except ImportError:
    pass