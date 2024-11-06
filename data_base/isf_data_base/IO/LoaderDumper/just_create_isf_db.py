"""Create and load :py:class:`~data_base.isf_data_base.isf-data_base.ISFDataBase` objects in a database.
"""

import os
# import cloudpickle
from . import parent_classes
import json

def check(obj):
    """Check whether the object can be saved with this dumper
    
    Args:
        obj (object): Object to be saved
    
    Returns:
        bool: Whether the object is None. This dumper requires no object to be saved.
    """
    return obj is None  #isinstance(obj, np) #basically everything can be saved with pickle


class Loader(parent_classes.Loader):
    """Loader for :py:class:`~data_base.isf_data_base.isf_data_base.ISFDataBase` objects"""
    def get(self, savedir):
        """Load the database from the specified folder"""
        return ISFDataBase(os.path.join(savedir, 'db'))


def dump(obj, savedir):
    """Create a :py:class:`~data_base.isf_data_base.isf_data_base.ISFDataBase` object in the specified :paramref:`savedir`
    
    Args:
        obj (None, optional): No object is required. If an object is passed, it is ignored.
        savedir (str): Directory where the database should be stored.
    """
    with open(os.path.join(savedir, 'Loader.json'), 'w') as f:
        json.dump({'Loader': __name__}, f)

from data_base.isf_data_base.isf_data_base import ISFDataBase