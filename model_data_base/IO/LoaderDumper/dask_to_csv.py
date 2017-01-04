import os
import cloudpickle
import dask.dataframe as dd
import dask.delayed
import pandas as pd
import parent_classes
import glob
from ... import settings


####################################################
# custom to_csv method, because the one of dask does eat all the memory
# this method has to the aim to be as simple as possible
#####################################################
def get_to_csv_function(index = None):
    def ddf_save_chunks(pdf, path, number, digits):
        pdf.to_csv(path.replace('*', str(number).zfill(digits)), index = index)
    return ddf_save_chunks
    
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

@dask.delayed
def bundle_delayeds(*args):
    pass

def my_to_csv(ddf, path, optimize_graph = False, index = None, get = settings.multiprocessing_scheduler):
    ''' Very simple method to store a dask dataframe to a bunch of csv files.
    The reason for it's creation is a lot of frustration with the respective 
    dask method, which has some weired hard-to-reproduce issues, e.g. it sometimes 
    takes all the ram (512GB!) or takes a very long time to "optimize" / merge the graph.
    '''
    
    ddf_save_chunks = get_to_csv_function(index = index)
    ddf = ddf.to_delayed()
    l = len(ddf)
    digits = len(str(l))
    save_delayeds = zip(ddf, [path]*l, list(range(l)), [digits]*l) #put all data together
    save_delayeds = map(dask.delayed(lambda x: ddf_save_chunks(*x)), save_delayeds) #call save function with it
    save_delayeds = bundle_delayeds(*save_delayeds) #bundle everything, so dask does not merge the graphs, which takes ages
    dask.compute(save_delayeds, optimize_graph = optimize_graph, get = get)
    
    

########################################################
# actual dumper
########################################################    
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
        my_read_csv = lambda x: pd.read_csv(x, index_col = self.index_name, skiprows = 1, \
                                                names = [self.index_name] + list(self.meta.columns))    
        if not self.index_name:
            print('loaded dask dataframe without index')
            ddf = dd.read_csv(os.path.join(savedir, fileglob))        
        elif self.index_name and self.divisions:
            print('loaded dask dataframe with index and known divisions')
            #it does not seem to be a good idea to pass the long index list through the delayed interface
            #therefore the list is contained in this function enclosure
            ddf = [dask.delayed(my_read_csv)(fname) \
                   for fname in sorted(glob.glob(os.path.join(savedir, fileglob)))]
            ddf = dd.from_delayed(ddf, divisions = self.divisions, meta = self.meta)
        elif self.index_name and not self.divisions:
            print('loaded dask dataframe with index but without known divisions')            
            ddf = [dask.delayed(my_read_csv)(fname) \
                   for fname in sorted(glob.glob(os.path.join(savedir, fileglob)))]
            ddf = dd.from_delayed(ddf, meta = self.meta)   
        return ddf
    
        
        
def dump(obj, savedir, repartition = False):
    if repartition:
        if obj.npartitions < 100:
            try:
                obj = obj.repartition(npartitions = 100)
            except ValueError:
                pass # can only repartition to fewer partitions ... will hopefully change in the future
        elif obj.npartitions >= 5000:
            obj = obj.repartition(npartitions = 5000)
            
    index_flag = obj.index.name is not None
    my_to_csv(obj, os.path.join(savedir, fileglob), get = settings.multiprocessing_scheduler, index = index_flag)
    #obj.to_csv(os.path.join(savedir, fileglob), get = settings.multiprocessing_scheduler, index = index_flag)
    meta = obj._meta
    index_name = obj.index.name
    if obj.known_divisions:
        divisions = obj.divisions
    else:
        divisions = None
        #experimental: calculate divisions, if they are not known and index is set
        if obj.index.name is not None:
            obj=obj.reset_index().set_index(index_name, sorted = True)
            divisions = obj.divisions
        
    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(meta, index_name, divisions), file_)