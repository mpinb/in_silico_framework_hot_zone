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

class MdbException(Exception):
    '''Typical mdb errors'''
    pass 

class MetadataAccessor:
    def __init__(self, mdb):
        self.mdb = mdb
        
    def __getitem__(self, key):
        dir_to_data = mdb._get_dir_to_data(key, check_exists = True)
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
    with open('/gpfs/soma_fs/scratch/abast/testmdbv2/1/metadata.json') as f:
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
        These elements are stored together with the other data in the basedir.
        All elements have associated metadata TODO: describe metadata
        
        They can be read out of the database in the following way:
        my_reloaded_element = mdb['my_new_element']
        
        It is possible to use tuples of strings as keys to reflect an arbitrary hierarchy.
        Valid keys are tuples of str or str. "@" is not allowed.
        
        To read out all existing keys, use the keys()-function.
        '''
        self.basedir = os.path.abspath(basedir)
        self.readonly = readonly
        self.nocreate = nocreate
        self.unique_id = None
        
        if not self._is_initialized():
            errstr = "Did not find a database in {path}. ".format(path = basedir) + \
                    "A new empty database will not be created since "+\
                    "{mode} is set to True."
            if nocreate:
                raise MdbException(errstr.format(mode = 'nocreate'))
            if readonly:
                raise MdbException(errstr.format(mode = 'readonly'))                
            self._initialize()
            
    def _register_this_database(self):
        print('registering database with unique id {} to the absolute path {}'.format(
            self.unique_id, self.basedir))
        try:
            model_data_base_register.register_mdb(self)
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
        # db_metadata.json may exist upon first init, but does not have a unique id yet. Create it and reset db_metadata
        time = os.stat(os.path.join(self.basedir, 'db_metadata.json')).st_mtime
        time = datetime.datetime.utcfromtimestamp(time)
        time = time.strftime("%Y-%m-%d")
        random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) 
                                for _ in range(7))
        self.unique_id = '_'.join([time, str(os.getpid()), random_string])

    def get_id(self):
        return self._unique_id 
     
    def _is_initialized(self):
        return os.path.exists(os.path.join(self.basedir, 'db_metadata.json'))
    
    def _initialize(self):
        _check_working_dir_clean_for_build(self.basedir)
        os.makedirs(self.basedir, exist_ok = True)
        # create empty metadata file. 
        with open(os.path.join(self.basedir, 'db_metadata.json'), 'w'):
            pass
        self._set_unique_id()
        self.save_db_metadata()
        
    def _check_key_format(self, key):
        assert(key != 'db_metadata.json')
    
        if len(key) > 50:
            raise ValueError('keys must be shorter than 50 characters')
        allowed_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_1234567890'
        for c in key:
            if not c in allowed_characters:
                raise ValueError('Character {} is not allowed'.format(c))        
        
    def _get_dir_to_data(self, key, check_exists = False):
        self._check_key_format(key)
        dir_to_data = os.path.join(self.basedir, key)
        if check_exists:
            if not os.path.exists(dir_to_data):
                raise KeyError('Key {} is not set.'.format(key))
        return dir_to_data
    
    def save_db_metadata(self):
        '''saves the data which defines the state of this database to db_metadata.json'''
        ## things that define the state of this mdb and should be saved
        out = {'registeredDumpers': self._registeredDumpers, \
               'unique_id': self.unique_id,
               'basedir': self.basedir} 
        with open(os.path.join(self.basedir, 'db_metadata.json'), 'w') as f:
            json.dump(out, f)

    def read_db_metadata(self):
        '''sets the state of the database according to dbcore.pickle''' 
        with open(os.path.join(self.basedir, 'db_metadata.json'), 'r') as f:
            out = json.load(f)
            
        for name in out:
            setattr(self, name, out[name])   
    
    def itemexists(self, key):
        '''Checks, if item is already in the database'''
        return key in list(self.keys())

    def get(self, key, lock = None, **kwargs):
        dir_to_data = self._get_dir_to_data(key, check_exists = True)
        # this looks into the metadat.json, gets the name of the dumper, and loads this module form IO.LoaderDumper
        loaderdumper_module = get_dumper_from_folder(dir_to_data)
        loader = loaderdumper_module.Loader()
        if lock:
            lock.acquire()
        return_ = loader.get(dir_to_data, **kwargs)
        if lock:
            lock.release()
        return return_
    
    def rename(self, old, new):
        dir_to_data_old = self._get_dir_to_data(old, check_exists = True)
        dir_to_data_new = self._get_dir_to_data(new)
        os.rename(old, new)

    def set(self, key, value, lock = None, dumper = None, **kwargs):
        assert(inspect.ismodule(dumper))
        dir_to_data = self._get_dir_to_data(key)
        if os.path.exists(dir_to_data):
            raise KeyError('Key {} is already set. Use del mdb[key] first.'.format(key))  
        else:
            os.makedirs(dir_to_data)
        loaderdumper_module = dumper
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
               'hostname': VC.get_hostname()}

        out.update(VC.get_git_version())

        if VC.get_git_version()['dirty']:
            warnings.warn('The database source folder has uncommitted changes!')
            
        with open(os.path.join(dir_to_data, 'metadata.json'), 'w') as f:
            json.dump(out, f)
            
    def setitem(self, key, value, dumper = None, **kwargs):
        warnings.warn('setitem is deprecated. it exist to provide a consistent API with model_data_base version 1. use set instead.')
        self.set(key, value, dumper = dumper, **kwargs)
    
    def __setitem__(self, key, value):
        self.set(key, value)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __delitem__(self, key):
        '''items can be deleted using del my_model_data_base[key]'''
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
     