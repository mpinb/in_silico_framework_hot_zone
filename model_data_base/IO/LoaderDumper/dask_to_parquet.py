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
def load_helper(savedir, n_partitions, partition):
    return pd.read_parquet(os.path.join(savedir, 'pandas_to_parquet.{}.{}.parquet'.format(n_partitions, partition)))

@dask.delayed
def save_helper(savedir, df, n_partitions, partition):
    return df.to_parquet(os.path.join(savedir, 'pandas_to_parquet.{}.{}.parquet'.format(n_partitions, partition)))

class Loader(parent_classes.Loader):
    def get(self, savedir):
        fnames = os.listdir(savedir)
        fnames = [f for f in fnames if 'pandas_to_parquet' in f]
        n_partitions = int(fnames[0].split('.')[1])
        delayeds = [load_helper(savedir, n_partitions, partition) for partition in range(n_partitions)]
        return dask.dataframe.from_delayed(delayeds)
    
def dump(obj, savedir, schema = None, client = None):
    delayeds = obj.to_delayed()
    delayeds = [save_helper(savedir, d, len(delayeds), lv) for lv, d in enumerate(delayeds)]
    futures = client.compute(delayeds)
    client.gather(futures)
    #obj.to_parquet(os.path.join(savedir, 'pandas_to_parquet.parquet'), schema = schema)
    compatibility.cloudpickle_fun(Loader(), os.path.join(savedir, 'Loader.pickle'))