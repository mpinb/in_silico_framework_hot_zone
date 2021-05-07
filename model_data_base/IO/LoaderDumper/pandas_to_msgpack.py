import os
# import cloudpickle
import compatibility
import pandas as pd
from . import parent_classes
import pandas_msgpack

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, (pd.DataFrame, pd.Series)) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def get(self, savedir):
#         return pd.read_msgpack(os.path.join(savedir, 'pandas_to_msgpack'))
        return pandas_msgpack.read_msgpack(os.path.join(savedir, 'pandas_to_msgpack'))
    
def dump(obj, savedir):
#     obj.to_msgpack(os.path.join(savedir, 'pandas_to_msgpack'), compress = 'blosc')
    pandas_msgpack.to_msgpack(os.path.join(savedir, 'pandas_to_msgpack'), obj, compress = 'blosc')

#     with open(os.path.join(savedir, 'Loader.pickle'), 'wb') as file_:
#         cloudpickle.dump(Loader(), file_)
    compatibility.cloudpickle_fun(Loader(), os.path.join(savedir, 'Loader.pickle'))
    

