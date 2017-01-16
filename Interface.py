import os
import sys
import tempfile
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd


from model_data_base import ModelDataBase
from model_data_base.analyze.burst_detection import burst_detection
from model_data_base.analyze.LDA import lda_prediction_rates as lda_prediction_rates
from model_data_base.analyze.temporal_binning import universal as temporal_binning
from model_data_base.analyze.spike_detection import spike_detection 
from model_data_base.analyze.spaciotemporal_binning import universal as spaciotemporal_binning

from model_data_base.IO.LoaderDumper import dask_to_csv as dumper_dask_to_csv
from model_data_base.IO.LoaderDumper import numpy_to_npy as dumper_numpy_to_npy
from model_data_base.IO.LoaderDumper import pandas_to_pickle as dumper_pandas_to_pickle

from model_data_base.IO.roberts_formats import write_pandas_synapse_activation_to_roberts_format
from model_data_base.IO.roberts_formats import read_pandas_synapse_activation_from_roberts_format

from model_data_base.mdb_initializers import load_crossing_over_results as mdb_init_crossing_over
from model_data_base.mdb_initializers import load_roberts_simulationdata as mdb_init_roberts_simulations

try: ##to avoid import errors in distributed system because of missing matplotlib backend
    from model_data_base.plotfunctions.average_std import average_std as average_std
    from model_data_base.plotfunctions.histogram import histogram as histogram
    from model_data_base.plotfunctions.manylines import manylines
    from model_data_base.plotfunctions.rasterplot import rasterplot
    from model_data_base.plotfunctions.cell_to_ipython_animation import cell_to_ipython_animation, display_animation
    from model_data_base.plotfunctions._figure_array_converter import show_pixel_object, PixelObject
    #PSTH = lambda x, **kwargs: histogram(temporal_binning(x), **kwargs)
    def PSTH(x, **kwargs):
        if 'groupby_attribute' in kwargs:
            assert('min_time' in kwargs)
            assert('max_time' in kwargs)        
            groupby_attribute = kwargs['groupby_attribute']
            del kwargs['groupby_attribute']
            colormap = kwargs['colormap']
            del kwargs['colormap']
            PSTH = x.groupby(groupby_attribute).apply(lambda x: temporal_binning(x, **kwargs))
            return histogram(PSTH, colormap = colormap, groupby_attribute = groupby_attribute)
        else:
            return histogram(temporal_binning(x), **kwargs)

    
except ImportError:
    pass

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

    
color_cellTypeColorMap = {'L1': 'cyan', 'L2': 'dodgerblue', 'L34': 'blue', 'L4py': 'palegreen',\
                    'L4sp': 'green', 'L4ss': 'lime', 'L5st': 'yellow', 'L5tt': 'orange',\
                    'L6cc': 'indigo', 'L6ccinv': 'violet', 'L6ct': 'magenta', 'VPM': 'black',\
                    'INH': 'grey', 'EXC': 'red', 'all': 'black', 'PSTH': 'blue'}

excitatory = ['L6cc', 'L2', 'VPM', 'L4py', 'L4ss', 'L4sp', 'L5st', 'L6ct', 'L34', 'L6ccinv', 'L5tt']
inhibitory = ['SymLocal1', 'SymLocal2', 'SymLocal3', 'SymLocal4', 'SymLocal5', 'SymLocal6', 'L45Sym', 'L1', 'L45Peak', 'L56Trans', 'L23Trans']

try:
    import single_cell_analyzer as sca
    import single_cell_parser as scp
except ImportError:
    pass