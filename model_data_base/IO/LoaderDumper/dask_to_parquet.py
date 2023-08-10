import os
# import cloudpickle
import compatibility
import pandas as pd
import dask
from . import parent_classes

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, dd.DataFrame)

@dask.delayed
def load_helper(path):
    return pd.read_parquet(os.path.join(savedir, 'pandas_to_parquet.parquet'))

@dask.delayed
def save_helper(path, df, partition):
    len_ = len(df)
    return pd.read_parquet(os.path.join(savedir, 'pandas_to_parquet.{}.{}.parquet'))

class Loader(parent_classes.Loader):
    def get(self, savedir):
        fnames = os.path.listdir(savedir)
        fnames = sorted(fnames, key = lambda x: int(x.split('.')[1]))
        fnames = [os.path.join(savedir, f) for f in fnames]
        delayeds = [load_helper(f) for f in fnames]
        return dask.dataframe.from_delayed(delayeds)
    
def dump(obj, savedir):
    obj.to_parquet(os.path.join(savedir, 'pandas_to_parquet.parquet'))
    compatibility.cloudpickle_fun(Loader(), os.path.join(savedir, 'Loader.pickle'))