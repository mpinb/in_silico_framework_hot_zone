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
from pathlib import Path
from . import _module_versions
VC = _module_versions.version_cached
from ._version import get_versions
from .IO.LoaderDumper import to_cloudpickle, just_create_folder, just_create_mdb_v2, shared_numpy_store, get_dumper_string_by_dumper_module
from . import model_data_base_v2_register
from . import MdbException
from .utils import calc_recursive_filetree, rename_for_deletion, delete_in_background, is_mdb, get_mdb_by_unique_id
from compatibility import pandas_unpickle_fun
import logging
logger = logging.getLogger("ISF").getChild(__name__)

DEFAULT_DUMPER = to_cloudpickle


class MetadataAccessor:
    """
    Access the metadata of some key
    It does not have a set method, as the metadata is set automatically when a key is set.
    Upon accidental metadata removal, the ModelDataBase will try to estimate the metadata itself using :func ModelDataBase._update_metadata_if_necessary:.
    """
    def __init__(self, mdb):
        self.mdb = mdb
        
    def __getitem__(self, key):
        key = self.mdb._convert_key_to_path(key)
        if not Path.exists(key/'metadata.json'):
            warnings.warn("No metadata found for key {}".format(key.name))
            return {
                'dumper': "unknown",
                'time': "unknown",
                'metadata_creation_time': 'post_hoc',
                'version': "unknown",
            }
        with open(key/'metadata.json') as f:
            return json.load(f)

    def keys(self):
        return [ k for k in self.mdb.keys() if Path.exists(self.mdb.basedir/k/"Loader.[json][pickle]") ]
        
def _check_working_dir_clean_for_build(working_dir):
    '''Backend method that checks, wether working_dir is suitable
    to build a new database there'''
    #todo: try to make dirs
    if Path.exists(working_dir):
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
    with open(Path(folder)/"metadata.json") as f:
        dumper_string = json.load(f)['dumper']
    if return_ == 'string':
        return dumper_string
    elif return_ == 'module':
        return importlib.import_module("model_data_base.IO.LoaderDumper.{}".format(dumper_string))

class ModelDataBase:
    def __init__(self, basedir, readonly = False, nocreate = False):
        '''
        Class responsible for robustly storing and retrieving information.
        It is meant to be used as an interface to simulation results. 
        If the dask backends are used to save the data, it will be out of memory, 
        allowing larger than memory calculations.
        
        Saved elements can be accessed using dictionary syntax:
        
        Example:
            my_reloaded_element = mdb['my_new_element']
        
        All saved elements are stored in the :arg basedir: along with metadata 
        and a Loader.pickle object. The Loader.pickle object contains which 
        module should be used to load the data with, along with all the necessary 
        information to initialize the Loader. This is done because some data 
        loaders need additional arguments
        
        All saved elements have associated metadata:
        - 'dumper': Which data dumper was used to save this result. 
            It's corresponding Loader can always be found in the same file. 
            See :module model_data_base.IO.LoaderDumper: for all dumpers and loaders.
        - 'time': Time at which this results was saved.
        - 'conda_list': A fill list of all modules installed in the conda environment 
            that was used to produce this result
        - 'module_versions': The versions of all modules in the conda environment 
            that was used to produce this result
        - 'history': The history of the code that was used to produce this result 
            in a Jupyter Notebook.
        - 'hostname': Name of the machine the code was run on.
        
        It is possible to use tuples of strings as keys to reflect an arbitrary hierarchy.
        Valid keys are tuples of str or str. "@" is not allowed.
        
        To read out all existing keys, use the keys() method.
        
        E.g. this class can be initialized in a way that after the initialization, 
        the data can be accessed in the following way:
        mdb['voltage_traces']
        mdb['synapse_activation']
        mdb['spike_times']
        mdb['metadata']
        mdb['cell_activation']
        
        Further more, it is possible to assign new elements to the database
        mdb['my_new_element'] = my_new_element

        Args:
            basedir (str): The directory in which the database will be created, or read from.
            readonly (bool, optional): If True, the database will be read only. Defaults to False.
            nocreate (bool, optional): If True, a new database will not be created if it does not exist. 
                Defaults to False.
        '''
        self.basedir = Path(basedir)
        self.readonly = readonly
        self.nocreate = nocreate
        self.parent_mdb = None

        # database state
        self._unique_id = None
        self._registeredDumpers = []
        self._registered_to_path = None

        self._forbidden_keys = [
            "dbcore.pickle", "metadata.db", "sqlitedict.db", "Loader.pickle",  # for backwards compatibility
            "db_state.json",
            "Loader.json"
        ]
        
        self.metadata = MetadataAccessor(self)
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
            self._infer_missing_metadata()  # In case some is missing

    def _infer_missing_metadata(self):
        '''
        Checks whether metadata is missing. Is so, it tries to estimate metadata, i.e. it sets the
        time based on the timestamp of the files. When metadata is created in that way,
        the field `metadata_creation_time` is set to `post_hoc`
        '''
        keys_in_mdb_without_metadata = set(self.keys()).difference(set(self.metadata.keys()))
        for key_str in keys_in_mdb_without_metadata:
            key = self._convert_key_to_path(key_str)
            print("Updating metadata for key {key}".format(key = key.name))
            try:
                dumper = LoaderDumper.get_dumper_string_by_savedir(key.as_posix())
            except FileNotFoundError:
                dumper = "unknown"
            
            time = os.stat(key).st_mtime
            time = datetime.datetime.utcfromtimestamp(time)
            time = tuple(time.timetuple())
            
            out = {
                'dumper': dumper, 
                'time': time,
                'metadata_creation_time': 'post_hoc',
                'version': 'unknown'
                }
            
            if VC.get_git_version()['dirty']:
                logging.warning('The database source folder has uncommitted changes!')
            # Save metdata, only for the key that does not have any
            json.dump(out, open(key/'metadata.json', 'w'))
            
    def _register_this_database(self):
        print('registering database with unique id {} to the absolute path {}'.format(
            self._unique_id, self.basedir))
        try:
            model_data_base_v2_register.register_mdb(self._unique_id, self.basedir)
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
        time = os.stat(self.basedir/'db_state.json').st_mtime
        time = datetime.datetime.utcfromtimestamp(time)
        time = time.strftime("%Y-%m-%d")
        random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) 
                                for _ in range(7))
        self._unique_id = '_'.join([time, str(os.getpid()), random_string])
    
    def _is_initialized(self):
        return Path.exists(self.basedir/'db_state.json')
    
    def _initialize(self):
        _check_working_dir_clean_for_build(self.basedir)   
        os.makedirs(self.basedir, exist_ok = True)
        # create empty state file. 
        with open(self.basedir/'db_state.json', 'w'):
            pass
        self._set_unique_id()
        self._registeredDumpers.append(DEFAULT_DUMPER)
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
            key (str|tuple): key
        """
        self._check_key_format(key)
           
    def _convert_key_to_path(self, key):
        self._check_key_format(key)
        if isinstance(key, str):
            return self.basedir/key
        elif isinstance(key, tuple):
            sub_mdb_path = ['mdb'] * (len(key) * 2 - 1)
            sub_mdb_path[0::2] = key
            return Path(self.basedir, *sub_mdb_path)
        else:
            assert isinstance(key, Path), "Key must be a string, tuple of strings, or Path. {} is type {}".format(key, type(key))
            return key
    
    def _check_key_format(self, key_str_tuple):
        """
        Checks the format of a key (string or tuple) and if it is valid for setting data (not for get).
        This is internal API and should never be called directly.
        This is the first line of checks when a user sets a key. For this reason, :arg key: is not Path, but a string or tuple.

        Args:
            key (str|tuple(str)): The key

        Raises:
            ValueError: If the key is over 50 characters long
            ValueError: If the key contains characters that are not allowed (only numeric or latin alphabetic characters, "-" and "_" are allowed)
        """
        assert isinstance(key_str_tuple, str) or isinstance(key_str_tuple, tuple), "Any key must be a string or tuple of strings. {} is type {}".format(key_str_tuple, type(key_str_tuple))
        if isinstance(key_str_tuple, str):
            key_str_tuple = key_str_tuple,  # convert to tuple
        assert all([isinstance(e, str) for e in key_str_tuple]), "Any key must be a string or tuple of strings. {} is type {}".format(key_str_tuple, type(key_str_tuple))

        # Check if individual characters are allowed
        for subkey in key_str_tuple:
            if len(subkey) > 50:
                raise ValueError('keys must be shorter than 50 characters')
            allowed_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_1234567890'
            for c in subkey:
                if not c in allowed_characters:
                    raise ValueError('Character {} is not allowed, but appears in key {}'.format(c, subkey))
        
    def _detect_dumper_string_of_existing_key(self, key):
        '''returns the dumper string of an existing key'''
        return get_dumper_from_folder(self._convert_key_to_path(key), return_ = 'string')
    
    def _find_dumper(self, item):
        '''
        Finds the dumper of a given item.
        Iterates all registered dumper and returns the first one that can save the item
        
        Args:
            item (object): The item to save

        Returns:
            module | None: The dumper module that can save the item, or None if there are no viable registered dumpers in the mdb.
        '''
        dumper = None
        for d in self._registeredDumpers:
            if d.check(item):
                dumper = d
                break
        return dumper
        
    def _write_metadata(self, dumper, dir_to_data):
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
            
        with open(dir_to_data/'metadata.json', 'w') as f:
            json.dump(out, f)
    
    def _check_writing_privilege(self, key):
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
    
    def check_if_key_exists(self, key):
        '''returns True, if key exists in a database, False otherwise'''
        return self._convert_key_to_path(key).exists()
    
    def get_id(self):
        return self._unique_id 
     
    def register_dumper(self, dumper_module):
        """
        Make sure to provide the module, not the class

        Args:
            dumper_module (module): A module from model_data_base.IO.LoaderDumper. Must contain a Loader class and a dump() method.
        
        """
        self._registered_dumpers.append(dumper_module)
    
    def save_db_state(self):
        '''saves the data which defines the state of this database to db_state.json'''
        ## things that define the state of this mdb and should be saved
        out = {'_registeredDumpers': [e.__name__ for e in self._registeredDumpers], \
               '_unique_id': self._unique_id,
               '_registered_to_path': self._registered_to_path.as_posix()} 
        with open(self.basedir/'db_state.json', 'w') as f:
            json.dump(out, f)

    def read_db_state(self):
        '''sets the state of the database according to dbcore.pickle''' 
        with open(self.basedir/'db_state.json', 'r') as f:
            state = json.load(f)
            
        for name in state:
            if name == '_registeredDumpers':
                # from string to module
                for dumper_string in state[name]:
                    self._registeredDumpers.append(importlib.import_module(dumper_string))
            else:
                setattr(self, name, state[name])

    def get_mkdtemp(self, prefix = '', suffix = ''):
        '''creates a directory in the model_data_base directory and 
        returns the path'''
        absolute_path = tempfile.mkdtemp(prefix = prefix + '_', suffix = '_' + suffix, dir = self.basedir) 
        os.chmod(absolute_path, 0o755)
        relative_path = absolute_path.relative_to(self.basedir)
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
    
    def create_sub_mdb(self, key, register = 'as_parent', **kwargs):
        '''creates a ModelDataBase within a ModelDataBase. Example:
        mdb.create_sub_mdb('my_sub_database')
        mdb['my_sub_database']['some_key'] = ['some_value']
        Kwargs will be passed to the dumper.

        Args:
            key (str|tuple): The key of the sub_mdb
            register (str, optional): ? TODO. Defaults to 'as_parent'.
            raise_ (bool, optional): Whether to raise an error if the sub_mdb already exists. Defaults to True.

        Kwargs:
            overwrite (bool, optional): Whether to overwrite the sub_mdb if it already exists. Defaults to True.
            Other kwargs will be passed to the dumper, and may depend on which dumper you're using. Consult :module model_data_base.IO.LoaderDumper: for more information on possible kwargs.

        Returns:
            ModelDataBase: The newly created sub_mdb
        '''
        self._check_key_format(key)
        if isinstance(key, str):
            key = key,  # convert to tuple
        # go down the tree of pre-existing sub_mdbs as long as the keys exist
        remaining_keys = key
        parent_mdb = self
        for i in range(len(key)):
            if key[i] not in parent_mdb.keys():
                # create sub_mdbs from here on
                break
            if not isinstance(parent_mdb[remaining_keys[0]], ModelDataBase):
                # The key exists and not an mdb, but we want to create one here
                raise MdbException(
                    "You were trying to overwrite existing data at %s with a (sub)mdb by using key %s. Please use del mdb[%s] first" % (
                        parent_mdb.basedir/key[i], key, key[:i+1]
                        ))
            # go down the tree of sub_mdbs
            parent_mdb = parent_mdb[remaining_keys[0]]
            remaining_keys = remaining_keys[1:]
        
        # If there are still unique keys remaining in the tuple, we have to create at least one sub_mdb
        for k in remaining_keys:
            parent_mdb.set(k, None, dumper = just_create_mdb_v2, **kwargs)
            # Note: registering this database happens upon initialization of the sub_mdb
            parent_mdb[k].parent_mdb = parent_mdb  # remember that it has a parent
            parent_mdb = parent_mdb[k]  # go down the tree of sub_mdbs
        # Either :arg raise_: is false and there are no remaining keys 
        #   -> simply return the pre-existing sub_mdb
        # or we just created it 
        #   -> return newly created sub_mdb
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
        key = self._convert_key_to_path(key)
        if not Path.exists(key):
            raise KeyError("Key {} not found in keys of mdb. Keys found: {}".format(key.name, self.keys()))
        if lock:
            lock.acquire()
        return_ = LoaderDumper.load(key, **kwargs)
        if lock:
            lock.release()
        return return_
    
    def rename(self, old_key, new_key):
        if not any([isinstance(e, str) or isinstance(e, Path) for e in [old_key, new_key]]):
            raise ValueError('old and new must be strings or Paths')
        old_key = Path(old_key)
        new_key = Path(new_key) 
        old_key.rename(new_key)

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
        if dumper is None:
            dumper = self._find_dumper(value)
        assert dumper is not None
        assert(inspect.ismodule(dumper))
        loaderdumper_module = dumper

        #check if we have writing privilege
        self._check_writing_privilege(key)
        self._check_key_format(key)

        # Use recursion to create sub_mdbs in case a tuple key is passed
        # All elements except for the last one should become sub_mdbs if they aren't already
        if isinstance(key, tuple) and len(key) > 1:
            sub_mdb = self.create_sub_mdb(key[0])  # create or fetch the sub_mdb
            # Recursion: set remaining subkeys in the sub_mdb
            sub_mdb.set(key[1:], value, lock = lock, dumper = dumper, **kwargs)
            return  # no further part of set() is needed for this subkey
        elif isinstance(key, tuple) and len(key) == 1:
            key = key[0]  # break recursion
        
        key = self._convert_key_to_path(key)

        # Key is Path and not a tuple if code made it here
        assert isinstance(key, Path)
        if Path.exists(key):  
            if is_mdb(key) or Path.exists(key/'mdb'):
                # Either the key is an mdb, or it's a key that contains a submdb
                # We are about to overwrite an mdb with data: that's a no-go (it's a me, Mario)
                raise MdbException(
                    "You were trying to overwrite a sub_mdb at %s with data using the key %s. Please remove the sub_mdb first, or use a different key." % (
                        self.basedir, key.name
                        )
                )
            # check if we can overwrite
            overwrite = kwargs.get('overwrite', True)  # overwrite=True if unspecified
            if overwrite:
                logger.info('Key {} is already set in ModelDatabase located at {}. Overwriting...'.format(key.name, self.basedir))
                delete_in_background(key)
            else:
                raise KeyError(
                    'Key {} is already set and you passed overwrite=False in the kwargs: {}'.format(key.name, kwargs) + \
                    '\nEither use del mdb[key] first, set overwrite to True, or omit the overwrite keyword argument.')  
        
        # Either the path does not exist yet, or it's in the process of being deleted
        os.makedirs(key)
        
        if lock:
            lock.acquire()
        try:
            loaderdumper_module.dump(value, key, **kwargs)
            self._write_metadata(dumper, key)
        except Exception as e:
            print("An error occured. Tidy up. Please do not interrupt.")
            try:
                shutil.rmtree(key)
            except:
                print('could not delete folder {:s}'.format(key.name))
            raise
        if lock:
            lock.release()
            
    def setitem(self, key, value, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.set(key, value, dumper = dumper, **kwargs)

    def getitem(self, key, lock=None, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.get(key, lock=lock, dumper=dumper, **kwargs)
    
    def maybe_calculate(self, key, fun, **kwargs):
        '''This function returns the corresponding value of key,
        if it is already in the database. If it is not in the database,
        it calculates the value by calling fun, adds this value to the
        database and returns the value.
        
        key: key on which the item can be accessed / should be accessible in the database
        fun: function expects no parameters (e.g. lambda: 'hello world') 
        force_calculation =: if set to True, the value will allways be recalculated
            If there is already an entry in the database with the same key, it will
            be overwritten
        **kwargs: attributes, that get passed to ModelDataBase.setitem
        
        Example:
        #value is calculated, since it is the first call and not in the database
        mdb.maybe_calculate('knok_knok', lambda: 'whos there?', dumper = 'self')
        > 'whos there?'
        
        #value is taken from the database, since it is already stored
        mdb.maybe_calculate('knok_knok', lambda: 'whos there?', dumper = 'self')
        > 'whos there?'        
        '''
        
        if 'force_calculation' in kwargs:
            force_calculation = kwargs['force_calculation']
            del kwargs['force_calculation']
        else:
            force_calculation = False
        try:
            if force_calculation:
                raise ValueError
            return self[key]
        except KeyError:
            ret = fun()
            self.setitem(key, ret, **kwargs)
            return ret    
    
    def keys(self):
        '''returns the keys of the database as string objects'''
        all_keys = self.basedir.iterdir()
        keys_ =  tuple(
            e.name for e in all_keys 
            if e.name not in ("db_state.json", "metadata.json", "Loader.json")
            and e.name not in ["dbcore.pickle", "metadata.db", "sqlitedict.db", "Loader.pickle"] # mdbv1 compatibility
            and ".deleting." not in e.name
            )
        return keys_

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
        to_delete = self._convert_key_to_path(key)
        delete_in_background(to_delete)
    
    def __reduce__(self):
        return (self.__class__, (self.basedir, self.readonly, True), {})

    def __repr__(self):
        return self._get_str()  # print with default depth and max_lines

    def print(self, depth=0, max_depth=2, max_lines=20, all_files=False, max_lines_per_key=3):
        """Prints out the content of the database in a tree structure.

        Args:
            max_depth (int, optional): How deep you want the filestructure to be. Defaults to 2.
            max_lines (int, optional): How long you want your filelist to be. Defaults to 20.
        """
        print(self._get_str(
            depth=depth, max_depth=max_depth, max_lines=max_lines, 
            all_files=all_files, max_lines_per_key=max_lines_per_key))
    
    def _get_str(self, depth=0, max_depth=2, max_lines=20, all_files=False, max_lines_per_key=3):
        """Fetches a string representation for this mdb in a tree structure.
        This is internal API and should never be called directly.
        
        Args:
            max_depth (int, optional): How deep you want the filestructure to be. Defaults to 2.
            max_lines (int, optional): How long you want your filelist to be. Defaults to 20.
            only_keys (bool, optional): Whether to only print keys only, or all files. Defaults to False.
            max_lines_per_key (int, optional): How many lines to print per key. Defaults to 4.

        Returns:
            str: A string representation of this mdb in a tree structure.

        """

        str_ = ['<{}.{} object at {}>'.format(self.__class__.__module__, self.__class__.__name__, hex(id(self)))]
        str_.append("Located at {}".format(self.basedir))
        # str_.append("{1}ModelDataBases{0} | {2}Directories{0} | {3}Keys{0}".format(
        #     bcolors.ENDC, bcolors.OKGREEN, bcolors.WARNING, bcolors.OKCYAN) )
        str_.append(bcolors.OKGREEN + self.basedir.name + bcolors.ENDC)
        lines = calc_recursive_filetree(
            self, Path(self.basedir), 
            depth=0, max_depth=max_depth, max_lines_per_key=max_lines_per_key, all_files=all_files)
        for line in lines:
            str_.append(line)
        return "\n".join(str_)

    def remove(self):
        '''
        Deletes the database from disk in the background and de-registers itself from the register as soon as it is deleted.
        Note that this method is not a destructor, nor equivalent to __del__ or __delete__.
        IOW, this method does not get called during garbage collection, when the object goes out of scope, or when the program terminates.
        It should be explicitly called by the user when the user likes to delete a database.
        '''
        def delete_and_deregister_once_deleted(dir_to_data, unique_id):
            shutil.rmtree(dir_to_data_rename)
            # this will delete in foreground of the thread, 
            # and thus wait until mdb is deleted and only then continue
            register = model_data_base_v2_register._get_mdb_register()
            del register[unique_id]  # remove from the register

        # make sure folder is renamed before continuing python process
        dir_to_data_rename = rename_for_deletion(self.basedir)
        # start processes on one thread in background
        threading.Thread(target = lambda : delete_and_deregister_once_deleted(self, self._unique_id)).start()

class RegisteredFolder(ModelDataBase):
    def __init__(self, path):
        ModelDataBase.__init__(self, path, forcecreate = True)
        self.setitem('self', None, dumper = just_create_folder)
        dumper = just_create_folder
        dumper.dump(None, path)
        self._sql_backend['self'] = LoaderWrapper('')
        self.setitem = None