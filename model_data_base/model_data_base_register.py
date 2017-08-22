import os
from .model_data_base import ModelDataBase, MdbException

_foldername = '.model_data_base_register'

class ModelDataBaseRegister():
    def __init__(self, basedir, search_mdbs = "on_first_init"):
        if not basedir.endswith(_foldername):
            basedir = os.path.join(basedir, _foldername)
        assert(basedir.endswith(_foldername))
        self.basedir = basedir
        self.mdb = ModelDataBase(basedir)
        if search_mdbs == "on_first_init" and self.mdb._first_init:
            self.search_mdbs(os.path.dirname(self.basedir))
        elif search_mdbs == True:
            self.search_mdbs(os.path.dirname(self.basedir))
    
    def search_mdbs(self, directory = None):
        for dir_ in [x[0] for x in os.walk(directory)]:
            if dir_.endswith(_foldername):
                continue
            try:
                mdb = ModelDataBase(dir_, readonly = True)
                self.mdb[mdb._unique_id] = os.path.abspath(dir_)
            except MdbException:
                continue
        
        print self.mdb.keys()
    
    def add_mdb(self, mdb):
        self.mdb[mdb._unique_id] = os.path.abspath(mdb.basedir)
        
def _get_mdb_register(dir_):
    dir_ = os.path.abspath(dir_)
    while True:
        path = os.path.join(dir_, _foldername)
        if os.path.exists(path):
            return ModelDataBaseRegister(path)
        dir_ = os.path.dirname(dir_)
        if dir_ == '/':
            raise MdbException("Did not find a ModelDataBaseRegister in parents of {path}"\
                               .format(path = dir_))
        
def register_mdb(mdb):
    mdbr = _get_mdb_register(mdb.basedir)
    mdbr.add_mdb(mdb)
    
def get_mdb_by_unique_id(parentdir_or_mdb, unique_id):
    if isinstance(parentdir_or_mdb, ModelDataBase):
        parentdir_or_mdb = parentdir_or_mdb.basedir
    mdbr = _get_mdb_register(parentdir_or_mdb)
    return ModelDataBase(mdbr.mdb[unique_id])
        

        
    