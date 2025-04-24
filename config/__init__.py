import os

def isf_is_using_mdb():
    """Check if the current environment is set to use :py:mod:`data_base.model_data_base`
    
    
    """
    return os.getenv("ISF_USE_MDB", 'False').lower() in ('true', '1', 't')