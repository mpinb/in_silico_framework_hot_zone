"""Configuration for ISF

This package provides ISF-wide configuration settings, such as `dask` memory overflow, file locking server configuration, logging configuration, cell types etc.
In general, these settings may change when switching hardware or animal species, but are unlikely to be varied otherwise.
"""

import os

def isf_is_using_mdb():
    """Check if ISF is configured to use :py:mod:`data_base.model_data_base`
    
    The use of :py:mod:`data_base.model_data_base` is strongly discouraged, as the saved data is not robust under API changes.
    
    There are two reasons to use it anyways:
    
    - Reading in existing data that has already been saved with this database system (i.e. the IBS Oberlaender Lab), in which case one must also `from ibs_projects import compatibility`
    - Testing purposes
    
    Returns:
        bool: whether or not ISF needs to use :py:mod:`data_base.model_data_base` as a database backend.
    """
    return os.getenv("ISF_USE_MDB", 'False').lower() in ('true', '1', 't')