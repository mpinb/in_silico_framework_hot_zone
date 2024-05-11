import os
# import cloudpickle
import compatibility
import pandas as pd
from . import parent_classes
from data_base.utils import df_colnames_to_str
import json
from .utils import save_object_meta, read_object_meta
import logging
logger = logging.getLogger("ISF").getChild(__name__)


def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(
        obj, (pd.DataFrame, pd.Series))


class Loader(parent_classes.Loader):

    def get(self, savedir):
        obj = pd.read_parquet(
            os.path.join(savedir, 'pandas_to_parquet.parquet'))
        try:
            # reset column dtype from string to original dtype.
            meta = read_object_meta(savedir)
            obj.columns = meta.columns
            obj.index = obj.index.astype(meta.index.dtype)
        except FileNotFoundError:
            logger.warning("No metadata found in {}\nColumn names and index will be string format".format(savedir))
        return obj
        


def dump(obj, savedir):
    save_object_meta(obj, savedir)
    # save original columns
    columns = obj.columns
    if obj.index.name is not None:
        index_name = obj.index.name
    # convert column names and index names to str
    # This overrides the original object, hence why we save the meta.
    obj = df_colnames_to_str(obj)
    # dump in parquet format
    obj.to_parquet(os.path.join(savedir, 'pandas_to_parquet.parquet'))
    with open(os.path.join(savedir, 'Loader.json'), 'w') as f:
        json.dump({'Loader': __name__}, f)
    # reset column names
    obj.columns = columns
    if obj.index.name is not None:
        obj.index.name = index_name
