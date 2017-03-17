import os
import cloudpickle
import dask.dataframe as dd
import dask.delayed
import pandas as pd
import parent_classes
import glob
from ... import settings

####
# if you want to use this as template to implement another dask dumper: 
# lines that most likely need to be changed are marked with ###
####

####################################################
# custom to_csv method, because the one of dask does eat all the memory
# this method has to the aim to be as simple as possible
#####################################################
def get_writer_function(index = None):
    '''returns function, that stores pandas dataframe'''
    def ddf_save_chunks(pdf, path, number, digits):
        pdf.to_msgpack(path.replace('*', str(number).zfill(digits)), compress = 'blosc') ###
    return ddf_save_chunks
    
def chunkIt(seq, num):
    '''makes approx. equal size chunks out of seq'''
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

@dask.delayed
def bundle_delayeds(*args):
    '''bundeling delayeds provided a huge speedup.
    this issue was adressed here:
    https://github.com/dask/dask/issues/1884
    '''
    pass

def my_dask_writer(ddf, path, optimize_graph = False, index = None, get = settings.multiprocessing_scheduler):
    ''' Very simple method to store a dask dataframe to a bunch of files.
    The reason for it's creation is a lot of frustration with the respective 
    dask method, which has some weired hard-to-reproduce issues, e.g. it sometimes 
    takes all the ram (512GB!) or takes a very long time to "optimize" / merge the graph.
    
    Update: this issue was addressed here:
    https://github.com/dask/dask/issues/1888
    '''
    
    ddf_save_chunks = get_writer_function(index = index)
    ddf = ddf.to_delayed()
    l = len(ddf)
    digits = len(str(l))
    save_delayeds = zip(ddf, [path]*l, list(range(l)), [digits]*l) #put all data together
    save_delayeds = map(dask.delayed(lambda x: ddf_save_chunks(*x)), save_delayeds) #call save function with it
    save_delayeds = bundle_delayeds(*save_delayeds) #bundle everything, so dask does not merge the graphs, which takes ages
    #dask.compute(save_delayeds, optimize_graph = optimize_graph, get = get)
    save_delayeds.compute(optimize_graph = True, get = get)
    

########################################################
# actual dumper
########################################################    
fileglob = 'dask_to_msgpack.*.csv' ###
def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, dd.DataFrame)

class Loader(parent_classes.Loader):
    def __init__(self, meta, index_name = None, divisions = None):        
        self.index_name = index_name
        self.meta = meta
        self.divisions = divisions
    def get(self, savedir, verbose = True):  
        #if dtypes is not defined (old mdb_versions) set it to None
        try:
            self.dtypes
        except AttributeError:
            self.dtypes = None
            
            
        my_reader = lambda x: pd.read_msgpack(x)  ###  
        
        if self.divisions:
            if verbose: print('loaded dask dataframe with known divisions')
            #it does not seem to be a good idea to pass the long index list through the delayed interface
            #therefore the list is contained in this function enclosure
            ddf = [dask.delayed(my_reader, traverse = False)(fname) \
                   for fname in sorted(glob.glob(os.path.join(savedir, fileglob)))]
            ddf = dd.from_delayed(ddf, divisions = self.divisions, meta = self.meta)
        else:
            if verbose: print('loaded dask dataframe without known divisions')            
            ddf = [dask.delayed(my_reader, tracerse = False)(fname) \
                   for fname in sorted(glob.glob(os.path.join(savedir, fileglob)))]
            ddf = dd.from_delayed(ddf, meta = self.meta)   
        ddf.index.name = self.index_name
        return ddf
    
        
        
def dump(obj, savedir, repartition = False):
    if repartition:
        if obj.npartitions > 10000:
            obj = obj.repartition(npartitions = 5000)
            
    index_flag = obj.index.name is not None
    my_dask_writer(obj, os.path.join(savedir, fileglob), get = settings.multiprocessing_scheduler, index = index_flag)
    #obj.to_csv(os.path.join(savedir, fileglob), get = settings.multiprocessing_scheduler, index = index_flag)
    meta = obj._meta
    index_name = obj.index.name
    if obj.known_divisions:
        divisions = obj.divisions
    else:
        divisions = None
        
    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(meta, index_name = index_name, divisions = divisions), file_)
        
        
        