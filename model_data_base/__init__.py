import logging

logger = logging.getLogger("ISF").getChild(__name__)

class MdbException(Exception):
    '''Typical mdb errors'''
    pass

class DataBaseException(Exception):
    '''Typical mdb errors'''
    pass

from .model_data_base import ModelDataBase
