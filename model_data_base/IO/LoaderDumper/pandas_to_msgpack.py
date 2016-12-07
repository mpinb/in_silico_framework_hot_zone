import os
import cloudpickle
import pandas as pd
import parent_classes

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, (pd.DataFrame, pd.Series)) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def get(self, savedir):
        return pd.read_msgpack(savedir)
    
def dump(obj, path):
    obj.to_msgpack(path)
        
    with open(os.path.join(path, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(), file_)
    

