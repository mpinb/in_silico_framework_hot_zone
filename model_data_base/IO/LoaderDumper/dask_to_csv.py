import os
import cloudpickle
import dask.dataframe as dd
import dask.delayed
import pandas as pd
import parent_classes
import glob
from ... import settings

fileglob = 'dask_to_csv.*.csv'
def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, dd.DataFrame) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def __init__(self, meta, index_name = None, divisions = None):
        self.index_name = index_name
        self.meta = meta

        self.divisions = divisions
    def get(self, savedir):      
        if not self.index_name:
            print('loaded dask dataframe without index')
            ddf = dd.read_csv(os.path.join(savedir, fileglob))        
        elif self.index_name and self.divisions:
            print('loaded dask dataframe with index and known divisions')
            ddf = [dask.delayed(pd.read_csv)(fname, index_col = self.index_name) \
                   for fname in glob.glob(os.path.join(savedir, fileglob))]
            ddf = dd.from_delayed(ddf, divisions = self.divisions, meta = self.meta)
        elif self.index_name and not self.divisions:
            print('loaded dask dataframe with index but without known divisions')            
            ddf = [dask.delayed(pd.read_csv)(fname, index_col = self.index_name) \
                   for fname in glob.glob(os.path.join(savedir, fileglob))]
            ddf = dd.from_delayed(ddf, meta = self.meta)   
        return ddf
        
def dump(obj, savedir):
    if obj.npartitions < 100:
        try:
            obj = obj.repartition(npartitions = 100)
        except ValueError:
            pass # can only repartition to fewer partitions ... will hopefully change in the future
    elif obj.npartitions >= 2000:
        obj = obj.repartition(npartitions = 2000)
    index_flag = obj.index.name is not None
    obj.to_csv(os.path.join(savedir, fileglob), get = settings.multiprocessing_scheduler, index = index_flag)
    meta = obj._meta
    index_name = obj.index.name
    if obj.known_divisions:
        divisions = obj.divisions
    else:
        divisions = None
        
    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(meta, index_name, divisions), file_)

