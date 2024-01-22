import logging

logger = logging.getLogger("ISF").getChild(__name__)

class DataBaseException(Exception):
    '''Typical isf_database errors'''
    pass

from .isf_data_base import DataBase
