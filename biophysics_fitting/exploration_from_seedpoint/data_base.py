"""
Wrapper class that decides whether or not a database is legacy ModelDataBase, or the new ISFDataBase.
"""
from data_base import model_data_base, isf_data_base
import sys
import os
from data_base.data_base_register import _get_db_register
import logging
logger = logging.getLogger('ISF').getChild(__name__)
sys.modules['data_base.IO'] = isf_data_base.IO

def is_data_base(path):
    """
    Checks if a given path is a ModelDataBase.
    """
    return os.path.exists(os.path.join(path, 'sqlitedict.db'))

class DataBase:
    def __new__(cls, basedir, readonly=False, nocreate=False):
        if is_data_base(basedir):
            logger.warning('Reading a legacy-format ModelDataBase.')
            return data_base.ModelDataBase(basedir, readonly=readonly, nocreate=nocreate)
        else:
            return isf_data_base.ISFDataBase(basedir, readonly=readonly, nocreate=nocreate)

def get_db_by_unique_id(unique_id):
    db_path = _get_db_register().registry[unique_id]
    db = DataBase(db_path, nocreate=True)
    assert db.get_id() == unique_id
    return db

