import os
import cloudpickle
import numpy as np
import parent_classes
from single_cell_parser.cell import Cell
from single_cell_parser.serialize_cell import save_cell_to_file
from single_cell_parser.serialize_cell import restore_cell_from_serializable_object

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return isinstance(obj, Cell) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def get(self, savedir):
        return restore_cell_from_serializable_object(os.path.join(savedir, 'cell'))
    
def dump(obj, savedir):
    save_cell_to_file(os.path.join(savedir, 'cell'), obj)

    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(), file_)