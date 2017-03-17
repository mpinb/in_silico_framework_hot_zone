import os
import cloudpickle
import parent_classes

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return obj is None #isinstance(obj, np) #basically everything can be saved with pickle

class Loader(parent_classes.Loader):
    def get(self, savedir):
        return savedir
    
def dump(obj, savedir):
    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(), file_)
    

