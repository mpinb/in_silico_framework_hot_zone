import os
import cloudpickle
import numpy as np
import parent_classes
import single_cell_parser as scp
from .serialize_cell import serialize_cell

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, np) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def get(self, savedir):
        return np.load(os.path.join(savedir, 'np.npy'))
    
def dump(obj, savedir):
    np.save(os.path.join(savedir, 'np.npy'), obj)

    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(), file_)
    

