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
from .IO.LoaderDumper import to_cloudpickle, just_create_folder, just_create_mdb_v2, shared_numpy_store, get_dumper_string_by_dumper_module
from . import model_data_base_v2_register
from . import MdbException
import logging
logger = logging.getLogger("ISF").getChild(__name__)

DEFAULT_DUMPER = to_cloudpickle


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
                'dumper': "unknown",
                'time': "unknown",
                'metadata_creation_time': 'post_hoc',
                'version': "unknown",
            }
        with open(os.path.join(dir_to_data, 'metadata.json')) as f:
            return json.load(f)

    def keys(self):
        return self.mdb.keys()
        
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
            self._update_metadata_if_necessary()

    def _update_metadata_if_necessary(self):
        '''
        checks whether metadata is missing. Is so, it tries to estimate metadata, i.e. it sets the
        time based on the timestamp of the files. When metadata is created in that way,
        the field `metadata_creation_time` is set to `post_hoc`
        '''
            
        keys_in_mdb_without_metadata = set(self.keys()).difference(set(self.metadata.keys()))
        for key in keys_in_mdb_without_metadata:
            print("Updating metadata for key {key}".format(key = str(key)))
            dir_to_data = mdb._get_dir_to_data(key)
            dumper = LoaderDumper.get_dumper_string_by_savedir(dir_to_data)
            
            time = os.stat(mdb._get_dumper_folder(key)).st_mtime
            time = datetime.datetime.utcfromtimestamp(time)
            time = tuple(time.timetuple())
            
            out = {
                'dumper': dumper, 
                'time': time,
                'metadata_creation_time': 'post_hoc'
                }
            
            if VC.get_git_version()['dirty']:
                warnings.warn('The database source folder has uncommitted changes!')
            
            mdb.metadata[key] = out
            
    def _register_this_database(self):
        print('registering database with unique id {} to the absolute path {}'.format(
            self._unique_id, self.basedir))
        try:
            model_data_base_v2_register.register_mdb(self._unique_id, self.basedir)
            self._registered_to_path = self.basedir
        except MdbException as e:
            warnings.warn(str(e))

    def _deregister_this_database(self):
        print('Deregistering database with unique id {} (had the absolute path {})'.format(
            self._unique_id, self.basedir))
        try:
            model_data_base_v2_register.deregister_mdb(self._unique_id)
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
    
    def _is_initialized(self):
        return os.path.exists(os.path.join(self.basedir, 'db_state.json'))
    
    def _initialize(self):
        _check_working_dir_clean_for_build(self.basedir)
        os.makedirs(self.basedir, exist_ok = True)
        # create empty state file. 
        with open(os.path.join(self.basedir, 'db_state.json'), 'w'):
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
        
       
    def _check_key_format(self, key):
        """
        Checks the format of a key (string or tuple) and if it is valid for setting data (not for get).
        This is internal API and should never be called directly.

        Args:
            key (str|tuple(str)): The key

        Raises:
            ValueError: If the key is over 50 characters long
            ValueError: If the key contains characters that are not allowed (only numeric or latin alphabetic characters, "-" and "_" are allowed)
        """
        def _is_sub_mdb(basedir, key):
            """
            Checks if some key points to a sub_mdb, without using methods like _get_dir_to_data (which use _check_key_validity).
            Checks if "metadata.json" exists within the key in the basedir.
            Does not check if basedir exists, or the key within the basedir.
            
            Checking key validity needs its own way to check if sub_mdbs are present, to avoid infinite recursion.
            This method is only used internally by _check_key_validity, and should never be called directly.

            Args:
                basedir (ModelDatabase): The 
                key (tuple): _description_
            """
            assert os.path.exists(basedir)
            # This should always exist
            dir_to_data = os.path.join(basedir, os.path.join(*key))
            assert os.path.exists(dir_to_data)
            return os.path.exists(os.path.join(dir_to_data, 'metadata.json'))


        assert isinstance(key, str) or isinstance(key, tuple), "Any key must be a string or tuple of strings. {} is type {}".format(key, type(key))
        assert all([isinstance(k, str) for k in key]), "Any key must be a string or tuple of strings. {} is type {}".format(key, type(key))
        key = tuple(key)

        # Check if individual characters are allowed
        for k in key:
            if len(k) > 50:
                raise ValueError('keys must be shorter than 50 characters')
            allowed_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_1234567890'
            for c in k:
                if not c in allowed_characters:
                    raise ValueError('Character {} is not allowed, but appears in key {}'.format(c, k))
        
        # check if all but last subkey of the key either points to a sub_mdb, 
        # or does not exist entirely (and sub_mdbs will be created)
        for k in [key[:i] for i in range(1, len(key))]:  # does not include last key
            dir_to_data = os.path.join(self.basedir, os.path.join(*k))
            if os.path.exists(dir_to_data) and not _is_sub_mdb(self.basedir, k):
                raise MdbException(
                    "Key {} points to a non-ModelDataBase, yet you are trying to save data to it with key {}.".format(k, key))
            else:
                # If a key in the tuple does not exist yet, sub_mdbs will be created
                break
        
        # check if the complete key refers to a value and not a sub_mdb
        # otherwise, an entire sub_mdb is about to be overwritten by data
        if os.path.exists(self._get_dir_to_data(key, check_format=False)) and _is_sub_mdb(self.basedir, key):
            raise MdbException(
                "Key {} points to a ModelDataBase, but you are trying to overwrite it with data. If you need this key for the data, please remove the sub_mdb under the same key first using del mdb[key] or mdb[key].remove()".format(key)) 
        
    def _get_dir_to_data(self, key, check_exists=False):
        '''returns the directory to the data of a given key
        
        Args:
            key (str|tuple(str)): The key
            check_exists (bool, optional): Whether to check if the key exists. Defaults to False. If True, a KeyError is raised if the key does not exist.
            check_format (bool, optional): Whether to check the format of the key. Defaults to True. If False, the key is not checked for validity, but only for existence.
        '''
        self._check_key_format(key)
        key = tuple(key)
        key_path = os.path.join(*key)
        dir_to_data = os.path.join(self.basedir, key_path)
        if check_exists:
            if not os.path.exists(dir_to_data):
                raise KeyError('Key {} is not set in mdb at directory {}.'.format(key_path, self.basedir))
        return dir_to_data

    def _detect_dumper_string_of_existing_key(self, key):
        '''returns the dumper string of an existing key'''
        dir_to_data = self._get_dir_to_data(key, check_exists = True)
        return get_dumper_from_folder(dir_to_data, return_ = 'string')
    
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
            
        with open(os.path.join(dir_to_data, 'metadata.json'), 'w') as f:
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
        try:
            self._get_dir_to_data(key, check_exists = True)
            return True
        except KeyError:
            return False
    
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
               '_registered_to_path': self._registered_to_path} 
        with open(os.path.join(self.basedir, 'db_state.json'), 'w') as f:
            json.dump(out, f)

    def read_db_state(self):
        '''sets the state of the database according to dbcore.pickle''' 
        with open(os.path.join(self.basedir, 'db_state.json'), 'r') as f:
            state = json.load(f)
            
        for name in state:
            if name == '_registeredDumpers':
                # from string to module
                for dumper_string in state[name]:
                    self._registeredDumpers.append(importlib.import_module(dumper_string))
            else:
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

    def create_sub_mdb(self, key, register = 'as_parent', **kwargs):
        '''creates a ModelDataBase within a ModelDataBase. Example:
        mdb.create_sub_mdb('my_sub_database')
        mdb['my_sub_database']['some_key'] = ['some_value']
        Kwargs will be passed to the dumper.

        TODO: don't override values with submdbs or vice-versa. Tried to make different key checks, but incomplete yet

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
        if isinstance(key, str):
            key = (key,)
        # go down the tree of pre-existing sub_mdbs as long as the keys exist
        remaining_keys = key
        parent_mdb = self
        while len(remaining_keys) > 0 and remaining_keys[0] in parent_mdb.keys():
            if not isinstance(parent_mdb[remaining_keys[0]], ModelDataBase):
                raise MdbException(
                    "Key %s is already set in %s and is not a ModelDataBase. Please use del mdb[%s] first" % (
                        remaining_keys[0], parent_mdb.basedir, remaining_keys[0]
                        ))
            # go down the tree
            parent_mdb = parent_mdb[remaining_keys[0]]
            # shift the remaining keys in the tuple
            remaining_keys = remaining_keys[1:]
        # If there are still unique keys remaining in the tuple, we have to create at least one sub_mdb
        for k in remaining_keys:
            parent_mdb._check_single_key_format(k)
            parent_mdb.set(k, None, dumper = just_create_mdb_v2, **kwargs)
            parent_mdb[k].parent_mdb = parent_mdb  # remember that it has a parent
            parent_mdb[k]._register_this_database()
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
        dir_to_data = self._get_dir_to_data(key, check_exists=True)
        if lock:
            lock.acquire()
        return_ = LoaderDumper.load(dir_to_data, **kwargs)
        if lock:
            lock.release()
        return return_
    
    def rename(self, old, new):
        dir_to_data_old = self._get_dir_to_data(old, check_exists = True)
        dir_to_data_new = self._get_dir_to_data(new)
        os.rename(old, new)

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

        # Use recursion to create sub_mdbs in case a tuple is passed
        # All elements except for the last one should become sub_mdbs
        # The last element should be saved in the last sub_mdb, and thus cannot be a submdb itself
        for key in tuple(key)[:-1]:
            sub_mdb = self.create_sub_mdb(key[0])  # create or fetch the sub_mdb
            # Recursion: keep making sub_mdbs until we reach the last element
            # at which point the for-loop will be skipped
            sub_mdb.set(key[1:], value, lock = lock, dumper = dumper, **kwargs)
            return  # don't continue after the for loop in case keys is still a tuple of size > 1
        
        # Key is not a tuple if code made it here
        dir_to_data = self._get_dir_to_data(key)
        if os.path.exists(dir_to_data):  # check if we can overwrite
            overwrite = kwargs.get('overwrite', True)  # overwrite=True if unspecified
            if overwrite:
                logger.info('Key {} is already set in ModelDatabase {} located at {}. Overwriting...'.format(key, self, self.basedir))
                delete_in_background(dir_to_data)
            else:
                raise KeyError(
                    'Key {} is already set and you passed overwrite=False in the kwargs: {}'.format(key, kwargs) + \
                    '\nEither use del mdb[key] first, set overwrite to True, or omit the overwrite keyword argument.')  
        
        # Either the path does not exist yet, or it's in the process of being deleted
        os.makedirs(dir_to_data)
        
        if lock:
            lock.acquire()
        try:
            loaderdumper_module.dump(value, dir_to_data, **kwargs)
            self._write_metadata(dumper, dir_to_data)
        except Exception as e:
            print("An error occured. Tidy up. Please do not interrupt.")
            try:
                shutil.rmtree(dir_to_data)
            except:
                print('could not delete folder {:s}'.format(dir_to_data))
            raise
        if lock:
            lock.release()
            
    def setitem(self, key, value, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.set(key, value, dumper = dumper, **kwargs)

    def getitem(self, key, lock=None, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.get(key, lock=lock, dumper=dumper, **kwargs)
    
    def keys(self):
        '''returns the keys of the database'''
        all_keys = os.listdir(self.basedir)
        keys_ =  tuple(
            e for e in all_keys 
            if e != "db_state.json"
            and e not in ["dbcore.pickle", "metadata.db", "sqlitedict.db"] # mdbv1 compatibility
            and not e.endswith(".lock") 
            and ".deleting." not in e
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
        dir_to_data = self._get_dir_to_data(key, check_exists = True) 
        delete_in_background(dir_to_data)
    
    def __reduce__(self):
        return (self.__class__, (self.basedir, self.readonly, True), {})

    def remove(self, deregister_timeout=3600):
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
            del _get_mdb_register()[unique_id]
        # make sure folder is renamed before continuing python process
        dir_to_data_rename = rename_for_deletion(self.basedir)
        # start processes on one thread in background
        threading.Thread(target = lambda : delete_and_deregister_once_deleted(self)).start()

class RegisteredFolder(ModelDataBase):
    def __init__(self, path):
        ModelDataBase.__init__(self, path, forcecreate = True)
        self.setitem('self', None, dumper = just_create_folder)
        dumper = just_create_folder
        dumper.dump(None, path)
        self._sql_backend['self'] = LoaderWrapper('')
        self.setitem = None
     

def get_mdb_by_unique_id(unique_id):
    mdb_path = model_data_base_v2_register._get_mdb_register().registry[unique_id]
    mdb = ModelDataBase(mdb_path, nocreate=True)
    assert mdb.get_id() == unique_id
    return mdb

def rename_for_deletion(dir_to_data):
    N = 5
    while True:
        random_string = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(N))
        dir_to_data_rename = dir_to_data + '.deleting.' + random_string
        if not os.path.exists(dir_to_data_rename):
            break
    os.rename(dir_to_data, dir_to_data_rename)
    return dir_to_data_rename

def delete_in_background(dir_to_data):
    dir_to_data_rename = rename_for_deletion(dir_to_data)
    p = threading.Thread(target = lambda : delete_from_disk(dir_to_data_rename)).start()
    return p


