# For backwards compatibility, register isf_data_base as top-level module, 
# so pickled data still knows where to find modules
import sys
sys.modules['isf_data_base'] = sys.modules[__name__]

class DataBaseException(Exception):
    '''Typical isf_database errors'''
    pass

from .isf_data_base import ISFDataBase
