import os
import cloudpickle
import parent_classes

def check(obj):
    '''checks wherther obj can be saved with this dumper'''
    return obj is None #isinstance(obj, np) #basically everything can be saved with pickle

class ManagedFolder(str):
    def __new__(cls, s, mdb):
        obj = str.__new__(cls, s)
        obj.mdb = mdb
        return obj
    def __init__(self, s, mdb):
        str.__init__(s)
        self.mdb = mdb
    def join(self, *args):
        return ManagedFolder(os.path.join(self, *args), self.mdb)

class Loader(parent_classes.Loader):
    def get(self, savedir):
        #return savedir
        return ManagedFolder(savedir, None)
    
def dump(obj, savedir):
    with open(os.path.join(savedir, 'Loader.pickle'), 'w') as file_:
        cloudpickle.dump(Loader(), file_)
    

