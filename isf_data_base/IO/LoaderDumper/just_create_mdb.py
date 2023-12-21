import os
# import cloudpickle
import compatibility
from . import parent_classes
import isf_data_base


def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return obj is None  #isinstance(obj, np) #basically everything can be saved with pickle


class Loader(parent_classes.Loader):

    def get(self, savedir):
        return isf_data_base.isf_data_base_legacy.DataBase(os.path.join(savedir, 'db'))


def dump(obj, savedir):
    #     with open(os.path.join(savedir, 'Loader.pickle'), 'wb') as file_:
    #         cloudpickle.dump(Loader(), file_)
    compatibility.cloudpickle_fun(Loader(),
                                  os.path.join(savedir, 'Loader.pickle'))
