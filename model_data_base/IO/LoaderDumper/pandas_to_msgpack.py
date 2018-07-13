import os
import cloudpickle
import pandas as pd
import parent_classes

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, (pd.DataFrame, pd.Series)) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def get(self, savedir):
        return pd.read_msgpack(os.path.join(savedir, 'pandas_to_msgpack'))
    
def dump(obj, savedir):
    obj.to_msgpack(os.path.join(savedir, 'pandas_to_msgpack'), compress = 'blosc')

    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(), file_)
    

