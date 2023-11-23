"""
Created October 2023

@authors: Arco Bast, Bjorge Meulemeester
"""

import os
import string
import warnings
import json
import threading
import random
import shutil
import inspect
import datetime
import importlib
from .IO import LoaderDumper
from . import _module_versions
VC = _module_versions.version_cached
from ._version import get_versions
from model_data_base.IO.LoaderDumper import to_cloudpickle, just_create_folder, just_create_mdb_v2, shared_numpy_store
from . import model_data_base_register

class MdbException(Exception):
    '''Typical mdb errors'''
    pass 

class MetadataAccessor:
    """Access the metadata of some key
    """
    def __init__(self, mdb):
        self.mdb = mdb
        
    def __getitem__(self, key):
        dir_to_data = self.mdb._get_dir_to_data(key, check_exists = True)
        if not os.path.exists(os.path.join(dir_to_data, 'metadata.json')):
            warnings.warn("No metadata found for key {}".format(key))
            return {
                'dumper': get_dumper_from_folder(dir_to_data),
                'time': "unknown",
                'metadata_creation_time': 'post_hoc',
                'version': "unknown",
            }
        with open(os.path.join(dir_to_data, 'metadata.json')) as f:
            return json.load(f)
        
def _check_working_dir_clean_for_build(working_dir):
    '''Backend method that checks, wether working_dir is suitable
    to build a new database there'''
    #todo: try to make dirs
    if os.path.exists(working_dir):
        try:
            if not os.listdir(working_dir):
                return
            else:
                raise OSError()
        except OSError:
            raise MdbException("Can't build database: " \
                               + "The specified working_dir is either not empty " \
                               + "or write permission is missing. The specified path is %s" % working_dir)
    else:
        try: 
            os.makedirs(working_dir)
            return
        except OSError:
            raise MdbException("Can't build database: " \
                               + "Cannot create the directories specified in %s" % working_dir)
            
    self.metadata = MetadataAccessor(self)

def make_all_str(dict_):
    out = {}
    for k,v in dict_.items():
        k = str(k)
        if isinstance(v, dict):
            out[k] = make_all_str(v)
        elif isinstance(v, str):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def get_dumper_from_folder(folder, return_ = 'module'):
    """Given a folder (i.e. key), return the dumper that was used to save the data in that folder/key.

    Args:
        folder (str): The folder in which the data is stored.
        return_ (str, optional): Whether to return the dumper as a string or the actual module. Defaults to 'module'.

    Returns:
        str | module: The dumper that was used to save the data in that folder/key.
    """
    with open(os.path.join(folder, "metadata.json")) as f:
        dumper_string = json.load(f)['dumper']
    if return_ == 'string':
        return dumper_string
    elif return_ == 'module':
        return importlib.import_module("model_data_base.IO.LoaderDumper.{}".format(dumper_string))

class ModelDataBase:
    def __init__(self, basedir, readonly = False, nocreate = False):
        '''
        Class responsible for storing information, meant to be used as an interface to simulation 
        results. If the dask backends are used to save the data, it will be out of memory,
        allowing larger than memory calculations.
        
        E.g. this class can be initialized in a way that after the initialization, 
        the data can be accessed in the following way:
        mdb['voltage_traces']
        mdb['synapse_activation']
        mdb['spike_times']
        mdb['metadata']
        mdb['cell_activation']
        
        Further more, it is possible to assign new elements to the database
        mdb['my_new_element'] = my_new_element
        
        All elements have associated metadata (see :class model_data_base._module_versions.Versions_cached:):
        - 'dumper': Which dumper was used to save this result. See :module model_data_base.IO.LoaderDumper: for available dumpers.
        - 'time': Time at which this results was saved.
        - 'conda_list': A fill list of all modules installed in the conda environment that was used to produce this result
        - 'module_versions': The versions of all modules in the conda environment that was used to produce this result
        - 'history': The history of the code that was used to produce this result in a Jupyter Notebook.
        - 'hostname': Name of the machine the code was run on.

        These elements are stored in the basedir along with metadata and a Loader.pickle object that allows it to be loaded in.
        
        They can be read out of the database in the following way:
        my_reloaded_element = mdb['my_new_element']
        
        It is possible to use tuples of strings as keys to reflect an arbitrary hierarchy.
        Valid keys are tuples of str or str. "@" is not allowed.
        
        To read out all existing keys, use the keys() method.

        Args:
            basedir (str): The directory in which the database will be created, or read from.
            readonly (bool, optional): If True, the database will be read only. Defaults to False.
            nocreate (bool, optional): If True, a new database will not be created if it does not exist. Defaults to False.
        '''
        self.basedir = os.path.abspath(basedir)
        self.readonly = readonly
        self.nocreate = nocreate
        self.parent_mdb = None

        # database state
        self._unique_id = None
        self._registeredDumpers = []
        self._registered_to_path = None
        
        if self._is_initialized():
            self.read_db_state()
        else:
            errstr = "Did not find a database in {path}. ".format(path = basedir) + \
            "A new empty database will not be created since "+\
            "{mode} is set to True."
            if nocreate:
                raise MdbException(errstr.format(mode = 'nocreate'))
            if readonly:
                raise MdbException(errstr.format(mode = 'readonly'))                
            self._initialize()
            
        if self.readonly == False:
            if self._unique_id is None:
                self._set_unique_id()
            if self._registered_to_path is None:
                self._register_this_database()
                self.save_db_state()
            
    def _register_this_database(self):
        print('registering database with unique id {} to the absolute path {}'.format(
            self._unique_id, self.basedir))
        try:
            model_data_base_v2_register.register_mdb(self)
            self._registered_to_path = self.basedir
        except MdbException as e:
            warnings.warn(str(e))
      
    def _set_unique_id(self):
        """
        Sets a unique ID for the model data base as class attribute. Does not save this ID as metadata (this is taken care of by :func _initialize:)

        Raises:
            ValueError: If the unique ID is already set.
        """
        if self._unique_id is not None:
            raise ValueError("self._unique_id is already set!")
        # db_state.json may exist upon first init, but does not have a unique id yet. Create it and reset db_state
        time = os.stat(os.path.join(self.basedir, 'db_state.json')).st_mtime
        time = datetime.datetime.utcfromtimestamp(time)
        time = time.strftime("%Y-%m-%d")
        random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) 
                                for _ in range(7))
        self._unique_id = '_'.join([time, str(os.getpid()), random_string])
    
    def get_id(self):
        return self._unique_id 
     
    def register_dumper(self, dumper_module):
        """
        Make sure to provide the module, not the class

        Args:
            dumper_module (module): A module from model_data_base.IO.LoaderDumper. Must contain a Loader class and a dump() method.
        
        """
        self._registered_dumpers.append(dumper_module)
    
    def _is_initialized(self):
        return os.path.exists(os.path.join(self.basedir, 'db_state.json'))
    
    def _initialize(self):
        _check_working_dir_clean_for_build(self.basedir)
        os.makedirs(self.basedir, exist_ok = True)
        # create empty state file. 
        with open(os.path.join(self.basedir, 'db_state.json'), 'w'):
            pass
        self._set_unique_id()
        self._registeredDumpers.append(to_cloudpickle)
        self.state = {
            '_registeredDumpers': self._registeredDumpers,
            '_unique_id': self._unique_id,
            '_registered_to_path': self._registered_to_path,
            }
            
        self._register_this_database()
        self.save_db_state()

    def _check_key_validity(self, key):
        """DEPRECATED! use _check_key_format instead.
        Only here for consistent API with mdbv1

        Args:
            key (str): key
        """
        self._check_key_format(key)
        
    def _check_key_format(self, key):    
        if len(key) > 50:
            raise ValueError('keys must be shorter than 50 characters')
        allowed_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_1234567890'
        for c in key:
            if not c in allowed_characters:
                raise ValueError('Character {} is not allowed'.format(c))  
        
    def _get_dir_to_data(self, key, check_exists = False):
        self._check_key_format(key)
        if isinstance(key, tuple):
            key = os.path.join(*key)
        dir_to_data = os.path.join(self.basedir, key)
        if check_exists:
            if not os.path.exists(dir_to_data):
                raise KeyError('Key {} is not set.'.format(key))
        return dir_to_data
    
    def _get_dumper_string(self, savedir, arg):
        path = self._get_path(arg)
        if path is None:
            return 'self'
        else:
            return IO.LoaderDumper.get_dumper_string_by_savedir(path)
    
    def save_db_state(self):
        '''saves the data which defines the state of this database to db_state.json'''
        ## things that define the state of this mdb and should be saved
        out = {'_registeredDumpers': [e.__name__ for e in self._registeredDumpers], \
               '_unique_id': self._unique_id,
               '_registered_to_path': self._registered_to_path} 
        with open(os.path.join(self.basedir, 'db_state.json'), 'w') as f:
            json.dump(out, f)

    def read_db_state(self):
        '''sets the state of the database according to dbcore.pickle''' 
        with open(os.path.join(self.basedir, 'db_state.json'), 'r') as f:
            state = json.load(f)
            
        for name in state:
            setattr(self, name, state[name])
    
    def itemexists(self, key):
        '''Checks, if item is already in the database'''
        return key in list(self.keys())

    def get_mkdtemp(self, prefix = '', suffix = ''):
        '''creates a directory in the model_data_base directory and 
        returns the path'''
        absolute_path = tempfile.mkdtemp(prefix = prefix + '_', suffix = '_' + suffix, dir = self.basedir) 
        os.chmod(absolute_path, 0o755)
        relative_path = os.path.relpath(absolute_path, self.basedir)
        return absolute_path, relative_path

    def create_managed_folder(self, key, raise_ = True):
        '''creates a folder in the mdb directory and saves the path in 'key'.
        You can delete the folder using del mdb[key]'''
        #todo: make sure that existing key will not be overwritten
        if key in list(self.keys()):
            if raise_:
                raise MdbException("Key %s is already set. Please use del mdb[%s] first" % (key, key))
        else:           
            self.setitem(key, None, dumper = just_create_folder)
        return self[key]

    def get_managed_folder(self, key):
        '''deprecated! Only here to have consistent API with mdb version 1.
        
        Use create_managed_folder instead'''   
        warnings.warn("Get_managed_folder is deprecated and only exists to have consistent API with mdbv1.  Use create_managed_folder instead.") 
        # TODO: remove this method
        return self.create_managed_folder(key)

    def create_shared_numpy_store(self, key, raise_ = True):
        if key in list(self.keys()):
            if raise_:
                raise MdbException("Key %s is already set. Please use del mdb[%s] first" % (key, key))
        else:
            self.setitem(key, None, dumper = shared_numpy_store)        
        return self[key]

    def create_sub_mdb(self, key, register = 'as_parent', raise_ = True):
        '''creates a ModelDataBase within a ModelDataBase. Example:
        mdb.create_sub_mdb('my_sub_database')
        mdb['my_sub_database']['some_key'] = ['some_value']
        '''
        if isinstance(key, str):
            key = (key,)
        # go down the tree of pre-existing sub_mdbs as long as the keys exist
        remaining_keys = key
        parent_mdb = self
        while key in parent_mdb.keys():
            parent_mdb = parent_mdb[key]
            remaining_keys = remaining_keys[1:]
        if not remaining_keys and raise_:
            # The sub_mdb already exists
            raise MdbException("Key %s is already set. Please use del mdb[%s] first" % (key, key)):
        for k in remaining_keys:
            parent_mdb._check_key_format(key)
            parent_mdb.set(key, None, dumper = just_create_mdb_v2)
            parent_mdb[key].parent_mdb = parent_mdb  # remember that it has a parent
            parent_mdb[key]._register_this_database()
            parent_mdb = parent_mdb[key]  # go down the tree of sub_mdbs
        # either raise is false and this mdb already exists, or we just created it
        return parent_mdb

    def get_sub_mdb(self,key, register = 'as_parent'):
        '''deprecated! it only exists to have consistent API to mdbv1
        
        Use create_sub_mdb instead'''
        warnings.warn("get_sub_mdb is deprecated. it only exists to have consistent API to mdbv1.  Use create_sub_mdb instead.")         
        #TODO: remove this method
        return self.create_sub_mdb(key, register = register)

    def get(self, key, lock = None, **kwargs):
        """This is the main method to get data from a ModelDataBase. :func getitem: and :func __getitem__: call this method.
        :func getitem: only exists to provide consistent API with mdbv1.
        :func __getitem__: is the method that's being called when you use mdb[key].
        The advantage is that this allows to pass additional arguments to the loader, e.g.
        mdb.getitem('key', columns = [1,2,3]).

        Args:
            key (str): the key to get from mdb[key]
            lock (Lock, optional): If you use file locking, provide the lock that grants access. Defaults to None.

        Returns:
            object: The object saved under mdb[key]
        """
        # this looks into the metadata.json, gets the name of the dumper, and loads this module form IO.LoaderDumper
        dir_to_data = self._get_dir_to_data(key, check_exists = True)
        loaderdumper_module = get_dumper_from_folder(dir_to_data)
        loader = loaderdumper_module.Loader()
        if lock:
            lock.acquire()
        return_ = loader.get(dir_to_data, **kwargs)
        if lock:
            lock.release()
        return return_
    
    def get_metadata(self, key, lock = None):
        """Given a kye, this method fetches the metedata associated with this key
        Apart from explicitly reading in the .json file using the absolute path, this is the
        only way to access the metadata of some key.

        Args:
            key (str): The key for which to fetch the metadata
            lock (Lock, optional): If using file locking. Defaults to None.

        Returns:
            _type_: _description_
        """
        dir_to_data = self._get_dir_to_data(key, check_exists = True)
        if lock:
            lock.acquire()
        return_ = MetadataAccessor(self)[key]
        if lock:
            lock.release()
        return return_
    
    def rename(self, old, new):
        dir_to_data_old = self._get_dir_to_data(old, check_exists = True)
        dir_to_data_new = self._get_dir_to_data(new)
        os.rename(old, new)

    def _find_dumper(self, item):
        '''finds the dumper of a given item.'''
        dumper = None
        for d in self._registeredDumpers:
            if d == 'self' or d.check(item):
                dumper = d
                break
        return dumper
    
    def set(self, key, value, lock = None, dumper = None, **kwargs):
        """Main method to save data in a ModelDataBase. :func setitem: and :func __setitem__: call this method.
        :func setitem: only exists to provide consistent API with mdbv1.
        :func __setitem__: is the method that's being called when you use mdb[key] = value.
        The advantage of using this method is that you can specify a dumper and pass additional arguments to the dumper with **kwargs.
        This method is thread safe, if you provide a lock.
        # TODO: deprecate the dumper "self". "self" only makes sense with an sqlite backend. "default" would be better in this case.

        Args:
            key (str): _description_
            value (obj): _description_
            lock (Lock, optional): _description_. Defaults to None.
            dumper (module|str|None, optional): The dumper module to use when saving data. If None or "self" are passed, it will use the default dumper to_cloudpickle. Defaults to None.

        Raises:
            KeyError: _description_
        """
        # Find correct dumper to save data with
        if dumper is None or dumper == 'self':
            dumper = self._find_dumper(value)
        assert dumper is not None
        assert(inspect.ismodule(dumper))
        loaderdumper_module = dumper

        # Check if the key is ok and create the corresponding path
        self._check_key_format(key)
        dir_to_data = self._get_dir_to_data(key)
        if os.path.exists(dir_to_data):
            raise KeyError('Key {} is already set. Use del mdb[key] first.'.format(key))  
        else:
            os.makedirs(dir_to_data)
        
        if lock:
            lock.acquire()
        try:
            loaderdumper_module.dump(value, dir_to_data, **kwargs)
            self._write_metadata(dumper, dir_to_data, key)
        except Exception as e:
            print("An error occured. Tidy up. Please do not interrupt.")
            try:
                shutil.rmtree(dir_to_data)
            except:
                print('could not delete folder {:s}'.format(basedir_absolute))
            raise
        if lock:
            lock.release()
        
    def _write_metadata(self, dumper, dir_to_data, key):
        '''this is private API and should only
        be called from within ModelDataBase.
        Can othervise be destructive!!!'''        
        dumper_string = LoaderDumper.get_dumper_string_by_dumper_module(dumper)

        out = {'dumper': dumper_string,
               'time': tuple(datetime.datetime.utcnow().timetuple()), 
               'conda_list': VC.get_conda_list(),
               'module_versions': make_all_str(VC.get_module_versions()),
               'history': VC.get_history(),
               'hostname': VC.get_hostname(),
               'metadata_creation_time': "together_with_new_key"}

        out.update(VC.get_git_version())

        if VC.get_git_version()['dirty']:
            warnings.warn('The database source folder has uncommitted changes!')
            
        with open(os.path.join(dir_to_data, 'metadata.json'), 'w') as f:
            json.dump(out, f)
            
    def setitem(self, key, value, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.set(key, value, dumper = dumper, **kwargs)

    def getitem(self, key, lock=None, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.get(key, lock=lock, dumper=dumper, **kwargs)
    
    def check_writing_privilege(self, key):
        '''raises MdbException, if we don't have permission to write to key '''
        if self.readonly is True:
            raise MdbException("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        #this exists, so jupyter notebooks will not crash when they try to write something
        elif self.readonly == 'warning': 
            warnings.warn("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        elif self.readonly == False:
            pass
        else:
            raise MdbException("Readonly attribute should be True, False or 'warning, but is: %s" % self.readonly)
    
    def keys(self):
        '''returns the keys of the database'''
        all_keys = os.listdir(self.basedir)
        keys =  tuple(
            e for e in all_keys 
            if e not in ["db_state.json", "dbcore.pickle", "metadata.db", "sqlitedict.db"]
            and "lock" not in e
            and "deleting" not in e)
        return keys

    def __setitem__(self, key, value):
        self.set(key, value)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __delitem__(self, key):
        """
        Items can be deleted using del my_model_data_base[key]
        Deleting an item will first rename the item to a random string and then delete it in the background.
        This way, your Python process is not interrupted when deleting large files, and you can immediately use the key again.
        
        """
        dir_to_data = self._get_dir_to_data(key, check_exists = True) 
        # rename folder to random folder
        N = 5
        while True:
            random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))
            dir_to_data_rename = dir_to_data + '.deleting.' + random_string
            if not os.path.exists(dir_to_data_rename):
                break
        os.rename(dir_to_data, dir_to_data_rename)
        threading.Thread(target = lambda : shutil.rmtree(dir_to_data_rename)).start()
    
    def __reduce__(self):
        return (self.__class__, (self.basedir, self.readonly, True), {})

class RegisteredFolder(ModelDataBase):
    def __init__(self, path):
        ModelDataBase.__init__(self, path, forcecreate = True)
        self.setitem('self', None, dumper = just_create_folder)
        dumper = just_create_folder
        dumper.dump(None, path)
        self._sql_backend['self'] = LoaderWrapper('')
        self.setitem = None
     