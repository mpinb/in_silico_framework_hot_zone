"""
Modular data base system switcher.

This module decides which data base system is used. it provides the :py:class:`~data_base.DataBase` class which initializes the correct
DataBase object depending on the provided path.

Newly created databases automatically use the newest data base system: :py:mod:`~data_base.isf_data_base`.
If the path points to a database that has been created with an older database system, this module returns the corresponding database object, with associated writers, readers, and file format readers.
"""
from .model_data_base.model_data_base import ModelDataBase
from .isf_data_base.isf_data_base import ISFDataBase
# For backwards compatibility, register model_data_base as top-level module, 
# so pickled data still knows where to find modules
import os
from .data_base_register import _get_db_register
import logging
logger = logging.getLogger('ISF').getChild(__name__)


class DataBase(object):
    def __new__(cls, basedir, readonly=False, nocreate=False):
        """
        As this is a wrapper class, it has no class attributes itself. Its reponsibility is to return the correct DataBase object.
        ModelDataBase is a deprecated class, and only used by the Oberlaender lab in Bonn for backwards compatibility with old data and Python 2.
        ISFDataBase is the new class, and should be used for all new data. This class is a wrapper that decides which class to return, so you don't have to remember.

        Args:
            basedir (str): The directory where the database is located.
            readonly (bool): If True, the database is read-only.
            nocreate (bool): If True, the database is not created if it does not exist.

        Returns
            ISFDataBase or ModelDataBase: The correct database object.
        """
        if is_model_data_base(basedir):
            # Allow to create mdbs during testing, otherwise read-only.
            nocreate = not os.environ.get('ISF_IS_TESTING', False)
            logger.warning('Reading a legacy-format ModelDataBase. nocreate is set to {}'.format(nocreate))
            logger.warning('Overwriting mdb.set and mdb.get to be compatible with ISF syntax...')
            
            db = ModelDataBase(basedir, readonly=readonly, nocreate=nocreate)
            db.set = db.setitem
            db.get = db.getitem
            db.create_sub_db = db.create_sub_mdb
            return db
        
        else:
            # return newest data base system
            return ISFDataBase(basedir, readonly=readonly, nocreate=nocreate)

def get_db_by_unique_id(unique_id):
    db_path = _get_db_register().registry[unique_id]
    db = DataBase(db_path, nocreate=True)
    assert db.get_id() == unique_id
    return db

def is_model_data_base(path):
    """
    Checks if a given path is a ModelDataBase, containing data that has been saved with ``model_data_base.IO.LoaderDumper.some_module``.
    """
    return os.path.exists(os.path.join(path, 'sqlitedict.db'))
