'''
Created on Aug 15, 2016

@author: arco
'''

import os, random, string
import shutil
import tempfile
import datetime
import cloudpickle as pickle
import IO
import IO.LoaderDumper.to_cloudpickle
import IO.LoaderDumper.pandas_to_msgpack
import IO.LoaderDumper.just_create_folder
import IO.LoaderDumper.just_create_mdb
from collections import defaultdict
# import model_data_base_register ## moved to end of file since this is a circular import


import analyze
import dask.diagnostics
import settings
from tuplecloudsqlitedict import SqliteDict
from copy import deepcopy
import warnings
import re
import inspect
from _version import get_versions


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    
    http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename
    a bit modified
    """
    import unicodedata
    value = str(value)
    value = unicode(value, errors = 'ignore')
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    value = unicode(re.sub('[^\w\s-]', '', value).strip().lower())
    value = unicode(re.sub('[-\s]+', '-', value))
    return str(value)
    
class LoaderWrapper:
    '''This is a pointer to data, which is stored elsewhere.
    
    It is used by ModelDataBase, if data is stored in a subfolder of the 
    model_data_base.basedir folder. It is not used, if the data is stored directly
    in the sqlite database.
    
    The process of storing data in a subfolder is as follows:
    1. The subfolder is generated using the mkdtemp method
    2. the respective dumper puts its data there
    3. the dumper also saves a Loader.pickle file there.
    4. A LoaderWrapper object pointing to the datafolder with a relative
        path (to allow moving of the database) is saved under the respective key
        in the model_data_base
        
    The process of loading in the data is as follows:
    1. the user request it: mdb['somekey']
    2. the result is a LoaderWrapper object
    3. the Loader.pickle file in the respective folder is loaded
    4. the get-methdo of the unpickled object is called with the
        absolute path to the folder.
    5. the returned object is returned to the user
    
    '''
    def __init__(self, relpath):
        self.relpath = relpath
        
class FunctionWrapper:
    '''This class wraps arouand an object that cannot be saved directly (e.g. 
    using pickle) but needs an initialization function instead.
    
    To use it, 
    1. create a function, that does not expect any argument and returns the object
        you want to save    
    2. create an instance of this class and pass your function
    3. save it in the database with wathever dumper you want'''
    def __init__(self, fun):
        warnings.warn("FunctionWrapper is deprecated!")
        self.fun = fun
                
class MdbException(Exception):
    '''Typical mdb errors'''
    pass        

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

###methods to hide dask progress bar based on settings:
import contextlib
@contextlib.contextmanager
def empty_context_manager(*args, **kwargs):
    '''does nothing. is meant to replace ProgressBar, if no output is needed'''
    yield

def get_progress_bar_function():
    if settings.show_computation_progress:
        PB = dask.diagnostics.ProgressBar
    else:
        PB = empty_context_manager
    return PB

class SQLBackend(object):
    def __init__(self, path):
        self.path = path
        
    def _get_sql(self):
        return SqliteDict(self.path, autocommit=True)
    
    def _direct_dbget(self, arg):
        '''Backend method to retrive item from the database'''
        try:
            sqllitedict = self._get_sql()
            dummy = sqllitedict[arg]
        finally:
            sqllitedict.close() 
        return dummy
    
    def _direct_dbset(self, key, item):
        '''Backend method to add a key-value pair to the sqlite database'''
        try:
            sqllitedict = self._get_sql()
            sqllitedict[key] = item
        except:
            raise
        finally: 
            sqllitedict.close()  

    def _direct_dbdel(self, arg):
        '''Backend method to delete item from the sqlite database.'''
        try:
            sqllitedict = self._get_sql()
            del sqllitedict[arg]
        finally:
            sqllitedict.close()      
    
    def keys(self):
        try:
            sqllitedict = self._get_sql()
            keys = sqllitedict.keys()
            return sorted(keys)
        finally:
            sqllitedict.close()                

class SQLMetadataAccessor():
    def __init__(self, sql_backend):
        self.sql_backend = sql_backend
        
    def __getitem__(self, key):
        out = defaultdict(lambda: 'unknown')
        if key in self.sql_backend.keys():
            out.update(self.sql_backend._direct_dbget(key))
        return out
    
    def keys(self):
        return self.sql_backend.keys()

class ModelDataBase(object):
    def __init__(self, basedir, forceload = False, readonly = False, nocreate = False):
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
        
        They can be read out of the database in the following way:
        my_reloaded_element = mdb['my_new_element']
        
        It is possible to use tuples of strings as keys to reflect an arbitrary hierarchy.
        Valid keys are tuples of str or str. "@" ist not allowed.
        
        To read out all existing keys, use the keys()-function.
        '''
                
        self.basedir = os.path.abspath(basedir)
        self.settings = settings #settings is imported above
        self.forceload = forceload
        self.readonly = readonly #possible values: False, True, 'warning'
        self._first_init = False
        self._unique_id = None
        
        try:
            self.read_db()
        except IOError:
            if nocreate:
                raise MdbException("Did not find a database in {path}. A new empty database will not be created since nocreate is set to True.")
            if readonly:
                raise MdbException("Did not find a database in {path}. A new empty database will not be created since readonly is set to True.")
            _check_working_dir_clean_for_build(basedir)
            self._first_init = True
            self._registeredDumpers = ['self'] #self: stores the data in the underlying database
            self.save_db()
            self._set_unique_id()
        
        self._sql_backend = SQLBackend(os.path.join(self.basedir, 'sqlitedict.db'))
        self._sql_metadata_backend = SQLBackend(os.path.join(self.basedir, 'metadata.db'))
        self.metadata = SQLMetadataAccessor(self._sql_metadata_backend)
        
        #############################
        # the following code helps to smoothly transient databases of the old
        # format (does not implement _unique_id and metadata) to the new format.
        # Should be commented out soon.
        ##############################
        if not self.readonly:
            self._update_metadata_if_necessary()        
            if self._unique_id is None:
                self._set_unique_id()
            self._register_this_database()

    def _register_this_database(self):
        try:
            model_data_base_register.register_mdb(self)
        except MdbException as e:
            warnings.warn(str(e))
            
    
    def _set_unique_id(self):
        if self._unique_id is not None:
            raise ValueError("self._unique_id is already set!")
        time = os.stat(os.path.join(self.basedir, 'dbcore.pickle')).st_mtime
        time = datetime.datetime.utcfromtimestamp(time)
        time = time.strftime("%Y-%m-%d")
        random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(7))
        self._unique_id = '_'.join([time, str(os.getpid()), random_string])
        self.save_db()      
        
    def get_id(self):
        return self._unique_id 
        
    def registerDumper(self, dumperModule):
        '''caveat: make sure to provide the MODULE, not the class'''
        self._registeredDumpers.append(dumperModule)
    
    def read_db(self):
        '''sets the state of the database according to dbcore.pickle''' 
        with open(os.path.join(self.basedir, 'dbcore.pickle'), 'r') as f:
            out = pickle.load(f)
            
        for name in out:
            setattr(self, name, out[name])            
            
    def save_db(self):
        '''saves the data which defines the state of this database to dbcore.pickle'''
        out = {'_registeredDumpers': self._registeredDumpers, \
               '_unique_id': self._unique_id} ## things that define the state of this mdb and should be saved
        with open(os.path.join(self.basedir, 'dbcore.pickle'), 'w') as f:
            pickle.dump(out, f)
        
    def itemexists(self, key):
        '''Checks, if item is already in the database'''
        return key in self.keys()
                
    def get_mkdtemp(self, prefix = '', suffix = ''):
        '''creates a directory in the model_data_base directory and 
        returns the path'''
        absolute_path = tempfile.mkdtemp(prefix = prefix + '_', suffix = '_' + suffix, dir = self.basedir) 
        relative_path = os.path.relpath(absolute_path, self.basedir)
        return absolute_path, relative_path

    def create_managed_folder(self, key):
        '''creates a folder in the mdb directory and saves the path in 'key'.
        You can delete the folder using del mdb[key]'''
        #todo: make sure that existing key will not be overwritten
        if key in self.keys():
                raise MdbException("Key %s is already set. Please use del mdb[%s] first" % (key, key))
        else:           
            self.setitem(key, None, dumper = IO.LoaderDumper.just_create_folder)
        return self[key]
        
    def get_managed_folder(self, key):
        '''deprecated!
        
        Use create_managed_folder instead'''   
        warnings.warn("Get_managed_folder is deprecated.  Use create_managed_folder instead.") 
        return self.create_managed_folder(key)
    
    def create_sub_mdb(self, key, register = 'as_parent'):
        '''creates a ModelDataBase within a ModelDataBase. Example:
        mdb.create_sub_mdb('my_sub_database')
        mdb['my_sub_database']['sme_key'] = ['some_value']
        '''
        if register == 'as_parent':
            ##todo
            pass
        if key in self.keys():
            raise MdbException("Key %s is already set. Please use del mdb[%s] first" % (key, key))
        else:
            self.setitem(key, None, dumper = IO.LoaderDumper.just_create_mdb)
        return self[key]
    
    def get_sub_mdb(self,key, register = 'as_parent'):
        '''deprecated!
        
        Use create_sub_mdb instead'''
        warnings.warn("get_sub_mdb is deprecated.  Use create_sub_mdb instead.")         
        return self.create_sub_mdb(key, register = register)
    
    def __getitem__(self, arg):
        '''items can be retrieved from the ModelDataBase using this syntax:
        item = my_model_data_base[key]'''
        dummy = self._sql_backend._direct_dbget(arg)
        if isinstance(dummy, LoaderWrapper):
            dummy = IO.LoaderDumper.load(os.path.join(self.basedir, dummy.relpath)) 
        if isinstance(dummy, FunctionWrapper):
            dummy = dummy.fun()
        return dummy     
    
    def setitem(self, key, item, **kwargs):
        '''Allows to set items. Compared to the mdb['some_keys'] = my_item syntax,
        this method allows more control over how the item is stored in the database.
        
        key: key
        item: item that should be saved
        dumper= dumper module to use, e.g. model_data_base.IO.LoaderDumper.numpy_to_npy
            If dumper is not set, the default dumper is used
        **kwargs: other keyword arguments that should be passed to the dumper
        '''
        
        
        #extract dumper from kwargs
        if not 'dumper' in kwargs:
            dumper = None
        else:
            dumper = kwargs['dumper']
            del kwargs['dumper']
            
            
        #check if we have writing privilege
        if self.readonly is True:
            raise MdbException("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        #this exists, so jupyter notebooks will not crash when they try to write something
        elif self.readonly is 'warning': 
            warnings.warn("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        elif self.readonly is False:
            pass
        else:
            raise MdbException("Readonly attribute is in unknown state. Should be True, False or 'warning, but is: %s" % self.readonly)
        
        #check if there is already a subdirectory assigned to this key. If so: store folder to delete it after new item is set.
        old_folder = None
        if key in self.keys():
            dummy = self._sql_backend._direct_dbget(key)
            if isinstance(dummy, LoaderWrapper):
                old_folder = dummy.relpath
                                
        #find dumper
        if dumper is None:
            for d in self._registeredDumpers:
                if d == 'self' or d.check(item):
                    dumper = d
                    break
    
        assert(dumper is not None)
                
        #if dumper is 'self': store in this DB
        if dumper == 'self':
            self._sql_backend._direct_dbset(key, item)
                
        #if dumper is something else: 
        #generate temp directory, save the object in that directory using dump()
        #wrap the relative path to an LoaderWrapper object and save it to the 
        #internal database
        else:
            basedir_absolute, basedir_relative = self.get_mkdtemp(prefix = slugify(key))
            try:
                dumper.dump(item, basedir_absolute, **kwargs)
                self._sql_backend._direct_dbset(key, LoaderWrapper(basedir_relative))
            except Exception as e:
                print("An error occured. Tidy up. Please do not interrupt.")
                try:
                    shutil.rmtree(basedir_absolute)
                except:
                    print 'could not delete folder %s' % basedir_absolute
                raise e
                
        
        if old_folder is not None:
            self._robust_rmtree(key, os.path.join(self.basedir, old_folder))
            
        #write metadata
        self._write_metadata_for_new_key(key, dumper)
    
    
    def _write_metadata_for_new_key(self, key, dumper):
        '''this is private API and should only
        be called from within ModelDataBase.
        Can othervise be destructive!!!'''        
        if inspect.ismodule(dumper):
            dumper = IO.LoaderDumper.get_dumper_string_by_dumper_module(dumper)
        elif isinstance(dumper, str):
            pass
        else:
            raise ValueError
        
        out = {'dumper': dumper, \
               'time': tuple(datetime.datetime.utcnow().timetuple()), \
               'metadata_creation_time': 'together_with_new_key'}
        
        out.update(get_versions())
        
        if get_versions()['dirty']:
            warnings.warn('The database source folder has uncommited changes!')
        
        self._sql_metadata_backend._direct_dbset(key, out)
    
    def _detect_dumper_string_of_existing_key(self, key):
        dumper = self._sql_backend._direct_dbget(key)
        if isinstance(dumper, LoaderWrapper):
            dumper = IO.LoaderDumper.get_dumper_string_by_savedir(os.path.join(self.basedir, dumper.relpath))
        else:
            dumper = 'self'
        return dumper
    
    def _get_dumper_folder(self, key):
        dumper = self._sql_backend._direct_dbget(key)
        if isinstance(dumper, LoaderWrapper):
            return os.path.join(self.basedir, dumper.relpath)
        else:
            return None        
            
    def _write_metadata_for_existing_key(self, key):
            '''this is private API and should only
            be called from within ModelDataBase.
            Can othervise be destructive!!!'''
            dumper = self._detect_dumper_string_of_existing_key(key)
            
            if dumper == 'self':
                time = 'unknown'
            else:
                time = os.stat(self._get_dumper_folder(key)).st_mtime
                time = datetime.datetime.utcfromtimestamp(time)
                time = tuple(time.timetuple())
            
            out = {'dumper': dumper, \
                   'time': time, \
                   'metadata_creation_time': 'post_hoc'}
            
            if get_versions()['dirty']:
                warnings.warn('The database source folder has uncommited changes!')
            
            self._sql_metadata_backend._direct_dbset(key, out)
    
    def _update_metadata_if_necessary(self):
        for key in self.keys():
            if key in self.metadata.keys():
                continue
            else:
                print "Updating metadata for key {key}".format(key = str(key))
                self._write_metadata_for_existing_key(key)
        
    def get_metadata(self, key):
        return self.metadata[key]
               
    def __setitem__(self, key, item):
        '''items can be set using my_model_data_base[key] = item
        This saves the data with the default dumper.
        
        A more elaborate version of this function, which allows more 
        control on how the data is stored in the database is 
        ModelDataBase.setitem.'''
                
        self.setitem(key, item, dumper = None)

    def _robust_rmtree(self, key, path):
        try:
            shutil.rmtree(path)
        except OSError:
            print('The folder ' + path + ' was registered as belonging to ' + \
                  str(key) + '. I tried to delete this folder, because the corresponding key was overwritten. ' + \
                  'Could not delete anything, because folder did not exist in the first place. I just carry on ...')
            
    def __delitem__(self, key):
        '''items can be deleted using del my_model_data_base[key]'''
        dummy = self._sql_backend._direct_dbget(key)
        if isinstance(dummy, LoaderWrapper):
            self._robust_rmtree(key, os.path.join(self.basedir,dummy.relpath))
        self._sql_backend._direct_dbdel(key)
                       
                
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
        except:
            with get_progress_bar_function()():
                ret = fun()
                self.setitem(key, ret, **kwargs)
            return ret    
        
    def keys(self):
        '''returns the keys of the database'''
        return self._sql_backend.keys()

    def __reduce__(self):
        return (self.__class__, (self.basedir, self.forceload, self.readonly, True))
    
import model_data_base_register


                      
        
    