from __future__ import absolute_import
import os
from .sqlite_backend.sqlite_backend import SQLiteBackend as SQLBackend
from .utils import cache
from .settings import model_data_base_register_path
from model_data_base import MdbException

_foldername = '.model_data_base_register.db'


class ModelDataBaseRegister():

    def __init__(self, registry_basedir, search_mdbs="on_first_init"):
        if not registry_basedir.endswith(_foldername):
            registry_basedir = os.path.join(registry_basedir, _foldername)
        assert registry_basedir.endswith(_foldername)
        self.registry_basedir = registry_basedir
        if not os.path.exists(self.registry_basedir):
            self._first_init = True
        else:
            self._first_init = False
        self.registry = SQLBackend(self.registry_basedir)
        if search_mdbs == "on_first_init" and self._first_init:
            self.search_mdbs(os.path.dirname(self.registry_basedir))
        elif search_mdbs == True:
            self.search_mdbs(os.path.dirname(self.registry_basedir))

    def search_mdbs(self, directory=None):
        for dir_ in [x[0] for x in os.walk(directory)]:
            if dir_.endswith(_foldername):
                continue
            if os.path.exists(os.path.join(dir_, "metadata.json")):
                # it is a model_data_base
                try:
                    with open(os.path.join(dir_, "metadata.json"), 'r') as f:
                        metadata = json.load(f)
                    self.add_mdb(metadata["unique_id"], dir_)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except MdbException:  # if there is no database
                    continue
                except Exception as e:
                    self.registry['failed', dir_] = e

    def add_mdb(self, unique_id, mdb_basedir):
        self.registry[mdb._unique_id] = os.path.abspath(mdb_basedir)


@cache
def _get_mdb_register():

    mdbr = ModelDataBaseRegister(model_data_base_register_path)
    return mdbr


#def _set_mdb_register(dir_):
#    global mdbr
#    mdbr = ModelDataBaseRegister(dir_)

#     dir_ = os.path.abspath(dir_)
#     while True:
#         path = os.path.join(dir_, _foldername)
#         #print path
#         if os.path.exists(path):
#             return ModelDataBaseRegister(path)
#         dir_ = os.path.dirname(dir_)
#         if dir_ == '/':
#             raise MdbException("Did not find a ModelDataBaseRegister.")


def register_mdb(unique_id, mdb_basedir):
    mdbr = _get_mdb_register()
    mdbr.add_mdb(unique_id, mdb_basedir)


# def get_mdb_by_unique_id(parentdir_or_mdb, unique_id):
#     if isinstance(parentdir_or_mdb, ModelDataBase):
#         parentdir_or_mdb = parentdir_or_mdb.basedir
#     mdbr = _get_mdb_register(parentdir_or_mdb)
#     return ModelDataBase(mdbr.mdb[unique_id])


def assimilate_remote_register(remote_path, local_path=_foldername):
    mdbr_remote = ModelDataBaseRegister(remote_path)
    mdbr_local = ModelDataBaseRegister(local_path)
    # get all remote model ids
    whole_registry = {k: mdbr_remote.registry[k] for k in mdbr_remote.registry.keys()}
    whole_registry_filtered = {
        k: v for k, v in whole_registry.items() if os.path.exists(v)
    }
    for k in whole_registry_filtered.keys():
        mdbr_local.registry[k] = whole_registry_filtered[k]
