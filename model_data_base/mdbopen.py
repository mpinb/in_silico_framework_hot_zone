from __future__ import absolute_import
import os
from .model_data_base import ModelDataBase, MdbException
from .model_data_base_register import get_mdb_by_unique_id
from .utils import cache

def resolve_mdb_path(path):
    # This has the purpose to map projects, that robert has run on the CIN cluster, to local paths
    if '/gpfs01/bethge/home/regger/data/' in path:
        print 'found CIN cluster prefix'
        print 'old path', path
        path = path.replace('/gpfs01/bethge/home/regger/data/', '/nas1/Data_regger/AXON_SAGA/Axon4/PassiveTouch/')
        print 'new path', path
    if not path.startswith('mdb://'):
        return path
    
    path_splitted = path.split('//')[1].split('/')

    try:
        mdb = get_mdb_by_unique_id(path_splitted[0])
    except KeyError:
        raise IOError("Trying to load {}. Did not find a ModelDataBase with id {}".format(path, path_splitted[0]))
    try:
        managed_folder = mdb[path_splitted[1]]
    except KeyError:
        raise KeyError("Trying to load {}. The Database has been found at {}. ".format(path, mdb.basedir) + \
        "However, this Database does not contain the key {}".format(path_splitted[1]))
    return os.path.join(managed_folder,*path_splitted[2:])


# def resolve_mdb_path(mdb, path):
#     path_splitted = path.split('::')
#     if not mdb.get_id() == path_splitted[0]:
#         try:
#             mdb = get_mdb_by_unique_id(mdb, path_splitted[0])
#         except KeyError:
#             raise IOError("Trying to load {}. Did not find a ModelDataBase with id {}".format(path, path_splitted[0]))
#     try:
#         managed_folder = mdb[path_splitted[1]]
#     except KeyError:
#         raise KeyError("Trying to load {}. The Database has been found at {}. ".format(path, mdb.basedir) + \
#         "However, this Database does not contain the key {}".format(path_splitted[1]))
#     return os.path.join(managed_folder,path_splitted[2])

@cache
def create_mdb_path(path):
    mdb_path = path
    while True:
        if (os.path.isdir(mdb_path)) and ('dbcore.pickle' in os.listdir(mdb_path)):
            break
        else:
            mdb_path = os.path.dirname(mdb_path)
        if mdb_path == '/':
            raise MdbException("The path {} does not seem to be within a ModelDatabase!".format(path))        
    mdb = ModelDataBase(mdb_path, nocreate = True)
        
    #print path
    path_minus_mdb_basedir = os.path.relpath(path, mdb.basedir)
    
    key = None
    for k in mdb.keys():
        v = mdb._sql_backend[k]
        try:
            if v.relpath == '': #this means, we have a RegisteredFolder class
                key = k
                break
            if v.relpath == path_minus_mdb_basedir.split('/')[0]:
                key = k
                break
        except AttributeError:
            pass
    
    if key is None:
        raise KeyError("Found a Database at {}. ".format(mdb.basedir)+\
                       "However, there is no key pointing to the subfolder {} in it."\
                       .format(path_minus_mdb_basedir.split('/')[0]))
    return os.path.join('mdb://', mdb.get_id(), key, os.path.relpath(path, mdb[key]))


class mdbopen:
    def __init__(self, path, mode = 'r'):
        self.path = path
        self.mode = mode
        
    def __enter__(self):
        self.path = resolve_mdb_path(self.path)
        self.f = open(self.path, self.mode)
        return self.f
    
    def __exit__(self, *args, **kwargs):
        self.f.close()
        
    