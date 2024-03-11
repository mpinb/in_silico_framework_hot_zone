import os
# import cloudpickle
import compatibility
from . import parent_classes
from data_base import isf_data_base
import json

def check(obj):
    '''checks whether obj can be saved with this dumper'''
    return obj is None  #isinstance(obj, np) #basically everything can be saved with pickle


class Loader(parent_classes.Loader):

    def get(self, savedir):
        return isf_data_base.isf_data_base.DataBase(os.path.join(savedir, 'db'))


def dump(obj, savedir):
    with open(os.path.join(savedir, 'Loader.json'), 'w') as f:
        json.dump({'Loader': __name__}, f)
