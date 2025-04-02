"""Re-optimize a database with a new dumper

Database optimization involves writing out various dataframes in a so-called `"optimized"` format.
It is sometimes of interest to re-write an already optimized database with a new (or old) data format.
For example, in the past we have switched from `msgpack` to `parquet`, and back to `msgpack` after un-deprecating it. 
So all databases optimized with `parquet` could now in principle be re-optimized with `msgpack`.
"""

from data_base.data_base import is_sub_data_base
from data_base.exceptions import DataBaseException
from .utils import _get_dumper
import shutil, os
import dask.dataframe as dd
import logging
from tqdm import tqdm

isf_logger = logging.getLogger("ISF")
logger = isf_logger.getChild(__name__)

DUMPERS_TO_REOPTIMIZE = [
    "pandas_to_parquet",
    "dask_to_parquet"
]

def _check_needs_reoptimization(key, old_dumper_name, new_dumper_name):
    if new_dumper_name == old_dumper_name:
        logger.warning("I am configured to re-optimize the dumper `{}`, but the current default optimized dumper for `{}` is also `{}`. Skipping this key...".format(
            old_dumper_name, key, new_dumper_name))
        return False
    else:
        logger.debug("Reoptimizing `{}` from `{}` to `{}`".format(key, old_dumper_name, new_dumper_name))
        return True

def _get_dumper_kwargs(d, client=None):
    """
    Get the dumper kwargs for a given DataFrame.
    This is used to determine if saving the dataframe requires a client or not.
    """
    if isinstance(d, dd.DataFrame):
        assert client is not None, "Please provide a dask client to re-optimize the database."
        return {"client": client}
    else:
        return {}
    

def _reoptimize_key(db, key, new_dumper, client=None):
    path_to_key = os.path.join(db.basedir, key)
    temp_key = key+"_reoptimizing"
    temp_path = os.path.join(db.basedir, temp_key)
        
    shutil.move(path_to_key, temp_path)  # move original key to tmp location
    
    try:
        d = db[temp_key]
        kwargs = _get_dumper_kwargs(d, client=client)
        db.set(key, d, dumper=new_dumper, **kwargs)
        shutil.rmtree(temp_path)
    except Exception as e:
        if os.path.exists(path_to_key):
            shutil.rmtree(path_to_key)  
        shutil.move(temp_path, path_to_key)  # move original key to tmp location
        raise DataBaseException(f"Failed to re-optimize {key}") from e


def reoptimize_db(db, client=None, progress=False, n_db_parents=0, suppress_warnings=False):
    """
    Re-optimize a sub-database with a new dumper.
    This is useful for switching between different data formats, such as from `parquet` to `msgpack`.
    """
    assert client is not None, "Please provide a dask client to re-optimize the database."
    logger.info("Reoptimizing database at {}".format(db.basedir))
    keys = db.keys() 
    if progress:
        if n_db_parents == 0: db_name = os.path.basename(db.basedir)
        else: db_name = os.path.basename(os.path.dirname(db.basedir))
        keys = tqdm(
            keys, 
            desc="Reoptimizing {}".format(db_name), 
            position=n_db_parents, 
            leave=False)
    original_level = isf_logger.level
    if suppress_warnings: isf_logger.setLevel(logging.ERROR)

    try:
        for key in keys:
            if is_sub_data_base(db, key):
                logger.info("Reoptimizing subdatabase {}".format(key))
                reoptimize_db(db[key], client=client, n_db_parents=n_db_parents+1, progress=progress)
                continue
            elif db.metadata[key]['dumper'] in DUMPERS_TO_REOPTIMIZE:
                is_categorizable = key in ("cell_activations", "synapse_activations")
                new_dumper = _get_dumper(db[key], categorized=is_categorizable)
                old_dumper_name = db.metadata[key]['dumper']
                
                if not _check_needs_reoptimization(key, old_dumper_name, new_dumper.__name__):
                    continue
                
                try:
                    _reoptimize_key(db, key, new_dumper, client=client)
                except Exception as e:
                    raise DataBaseException(f"Failed to re-optimize {key}") from e
    finally:
        if suppress_warnings:
            isf_logger.setLevel(original_level)
