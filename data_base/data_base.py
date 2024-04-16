"""
Wrapper class that decides whether or not a database is legacy ModelDataBase, or the new ISFDataBase.
"""
from data_base import model_data_base, isf_data_base
import sys
# For backwards compatibility, register model_data_base as top-level module, 
# so pickled data still knows where to find modules
sys.modules['model_data_base'] = model_data_base
import os
from data_base.data_base_register import _get_db_register
import logging
logger = logging.getLogger('ISF').getChild(__name__)


class DataBase:
    def __new__(cls, basedir, readonly=False, nocreate=False):
        if is_model_data_base(basedir):
            logger.warning('Reading a legacy-format ModelDataBase.')
            db = model_data_base.ModelDataBase(basedir, readonly=readonly, nocreate="True")
            logger.warning('Overwriting mdb.set and mdb.get to be compatible with ISF syntax...')
            db.set = db.setitem
            db.get = db.getitem
            return db
        else:
            return isf_data_base.ISFDataBase(basedir, readonly=readonly, nocreate=nocreate)

def get_db_by_unique_id(unique_id):
    db_path = _get_db_register().registry[unique_id]
    db = DataBase(db_path, nocreate=True)
    assert db.get_id() == unique_id
    return db

def is_model_data_base(path):
    """
    Checks if a given path is a ModelDataBase, containing data that has been saved with model_data_base.IO.LoaderDumper.some_module.
    """
    return os.path.exists(os.path.join(path, 'sqlitedict.db'))
