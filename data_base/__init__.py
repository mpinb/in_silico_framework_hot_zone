"""
This package provides an efficient, scalable, and flexible database whose API mimics a python dictionary. 
It provides efficient and scalable methods to store and access simulation results at a terrabyte scale. 
It also generates metadata, indicating when the data was put in the database and the exact version of the in_silico_framework that was used at this timepoint. 
Simulation results from the :py:mod:`single_cell_parser` module can be imported and converted to a high performance binary format. 
Afterwards the data is accessible using the pandas data analysis library and dask.

It contains two important subpackages:

    1. :py:mod:model_data_base: A legacy format, only used by the Oberlaender lab at In-Silico Brain Sciences, MPINB Bonn. This should never be used by anyone else.
    2. :py:mod:isf_data_base: An updated data_base package, using JSON as metadata format, and the newest file formats, such as parquet.
"""
import sys
import isf_data_base.IO
sys.modules['data_base.IO'] = isf_data_base.IO

class DataBaseException(Exception):
    '''Typical data_base errors'''
    pass