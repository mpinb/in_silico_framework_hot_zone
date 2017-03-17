'''
This has the only purpose to make the simrun2 and the model_data_base package
more convenient.

It therefore imports all the "important" functions from those modules,
so that they are directly available. It also contains some small 
functions, that combine two functions, that are frequently used together,
e.g. instead of displaying a PSTH, you can simply use the PSTH function
which combines binning and displaying in one function.

A recommendet use is to import it in a jupyter notebook in the following manner:
    import Interface as I
    
'''

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
from model_data_base.analyze import split_synapse_activation
from model_data_base.analyze.analyze_input_mapper_result import compare_to_neuronet

from model_data_base.IO.LoaderDumper import dask_to_csv as dumper_dask_to_csv
from model_data_base.IO.LoaderDumper import numpy_to_npy as dumper_numpy_to_npy
from model_data_base.IO.LoaderDumper import pandas_to_pickle as dumper_pandas_to_pickle
from model_data_base.IO.LoaderDumper import dask_to_msgpack as dumper_dask_to_msgpack
from model_data_base.IO.LoaderDumper import dask_to_categorized_msgpack as dumper_dask_to_categorized_msgpack

#from model_data_base.IO.LoaderDumper import just_create_folder as dumper_just_create_folder
from model_data_base.IO.LoaderDumper import cell as dumper_cell



from model_data_base.IO.roberts_formats import write_pandas_synapse_activation_to_roberts_format
from model_data_base.IO.roberts_formats import read_pandas_synapse_activation_from_roberts_format
from model_data_base.IO.roberts_formats import read_pandas_synapse_activation_from_roberts_format
from model_data_base.IO.roberts_formats import read_InputMapper_summary

from model_data_base.mdb_initializers import load_crossing_over_results as mdb_init_crossing_over
from model_data_base.mdb_initializers import load_roberts_simulationdata as mdb_init_roberts_simulations
from model_data_base.mdb_initializers import load_roberts_simulationdata2 as mdb_init_simrun_general


from model_data_base.analyze import split_synapse_activation, color_cellTypeColorMap, excitatory, inhibitory
from model_data_base.utils import silence_stdout
from model_data_base.utils import select, pandas_to_array, pooled_std
from model_data_base.utils import skit, chunkIt

def split_synapse_activation(sa, selfcheck = True, excitatory = excitatory, inhibiotry = inhibitory):
    '''Splits synapse activation in EXC and INH component.
    
    Assumes, that if the cell type mentioned in the column synapse_type is
    in the list Interface.excitatory, that the synapse is excitatory, else inhibitory.
    
    selfcheck: Default: True. If True, it is checked that every celltype is either
    asigned excitatory or inhibitory
    '''
    if selfcheck:
        celltypes = sa.apply(lambda x: x.synapse_type.split('_')[0], axis = 1).drop_duplicates()
        for celltype in celltypes:
            assert(celltype in excitatory + inhibitory)
            
    sa['EI'] = sa.apply(lambda x: 'EXC' if x.synapse_type.split('_')[0] in excitatory else 'INH', axis = 1)
    return sa[sa.EI == 'EXC'], sa[sa.EI == 'INH']

try: ##to avoid import errors in distributed system because of missing matplotlib backend
    from model_data_base.plotfunctions.average_std import average_std as average_std
    from model_data_base.plotfunctions.histogram import histogram as histogram
    from model_data_base.plotfunctions.manylines import manylines
    from model_data_base.plotfunctions.rasterplot import rasterplot
    from model_data_base.plotfunctions.cell_to_ipython_animation import cell_to_ipython_animation, display_animation
    from model_data_base.plotfunctions._figure_array_converter import show_pixel_object, PixelObject

    def PSTH(x, **kwargs):
        '''Combines the temporal_binning function with the histogram function.'''
        kwargs_temporal_binning, kwargs_histogram = skit(temporal_binning, histogram, **kwargs)
        print kwargs
        bins = temporal_binning(x, **kwargs_temporal_binning)
        return histogram(bins, **kwargs_histogram)
    PSTH.__doc__ = PSTH.__doc__ + \
                '\n\nDocs temporal_binning:' + str(temporal_binning.__doc__) + \
                '\n\nDocs histogram: ' + str(histogram.__doc__)

    def PSTH_spaciotemporal(df, distance_column = 'soma_distance', **kwargs):
        '''Combines the spaciotemporal_binning function with the histogram function.
        
        kwargs: colormap and groupby_attribute will be forwarded to histogram.
        All other kwargs are forwarded to temporal binning.
        
        kwargs: 
        spacial_distance_bins = 50, 
        min_time = 0, \
        max_time = 300, time_distance_bins = 1
        '''
        
        if 'distance_column' in kwargs:
            distance_column = kwargs['distance_column']
        else: 
            kwargs['distance_column'] = 'soma_distance'
        
        bins = spaciotemporal_binning(df, **kwargs)
    
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
    
from simrun2 import crossing_over as simrun_crossing_over_module
from simrun2.crossing_over.crossing_over_simple_interface import crossing_over as simrun_crossing_over_simple_interface


try:
    import single_cell_analyzer as sca
    import single_cell_parser as scp
except ImportError:
    pass