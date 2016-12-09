'''
Created on Aug 15, 2016

@author: arco
'''

import os
import tempfile
import cloudpickle as pickle
import IO, IO.LoaderDumper.to_pickle, analyze
import dask.diagnostics
import settings
from tuplecloudsqlitedict import SqliteDict
from copy import deepcopy
import warnings

# #monkey patch pandas to provide parallel computing methods
# def apply_parallel(self, *args, **kwargs):
#     '''Takes a pandas dataframe, converts it to a dask dataframe, applies the given method in parallel,
#     and converts the result back to a pandas dataframe. Introduces a lot of overhead but is convenient.
#     '''
#     #Note: npartitions = 80 seems to be a good choice if one of the compute servers with 40 cores is used.
#     #Otherwise, it might be necessary to change that value
#     return dask.dataframe.from_pandas(self, npartitions = 80).apply(*args, **kwargs).compute(get = dask.multiprocessing.get)
# pd.DataFrame.apply_parallel = apply_parallel

class LoaderWrapper:
    def __init__(self, relpath):
        self.relpath = relpath
        
class MdbException(Exception):
    pass        

def _check_working_dir_clean_for_build(working_dir):
    #todo: try to make dirs
    if os.path.exists(working_dir):
        try:
            os.rmdir(working_dir) #only works, if dir was empty --- but is it save?
            os.mkdir(working_dir)
            return
        except OSError:
            raise MdbException("Can't build database: " \
                               + "The specified working_dir is either not empty " \
                               + "or write permission is missing.")
    else:
        try: 
            os.makedirs(working_dir) #todo: should ideally be mkdirs
            return
        except OSError:
            raise MdbException("Can't build database: " \
                               + "Cannot create the directories specified in working_dir.")

class ModelDataBase(object):
    def __init__(self, basedir, forceload = False, readonly = False):
        '''
        Class responsible for storing information, meant to be used as an interface to simulation 
        results. 
        
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
        
        They can be re ad out of the database in the following way:
        my_reloaded_element = mdb['my_new_element']
        
        It is possible to use tuples of strings as keys to reflect an arbitrary hierarchy.
        To read out all existing keys, use the keys()-function.
        
        After an update of the underlying libraries (e.g. dask), it might be,
        that the standard data ('voltage_traces', 'synapse_activation' and so on) 
        can not be unpickled any more. In this case, you can rebuild these
        Dataframes by calling the _regenerate_data() method. If you regain
        access to the underlying data e.g. mdb['voltage_traces'], you should
        consider to save the updated database with the save_db() method.     
        '''
        
        #todo: rename basedir to basedir
        
        self.basedir = basedir
        self.settings = settings #settings is imported above
        self.forceload = forceload
        self.readonly = readonly #possible values: False, True, 'warning'
        
        try:
            self.read_db()
        except:
            _check_working_dir_clean_for_build(basedir)
            self.first_init()
            self.save_db()
            
    def first_init(self):
        '''function to initialize this db with default values, if it has never been
        initialized before'''
        self._registeredDumpers = ['self'] #self: stores the data in the underlying database
        
    def registerDumper(self, dumperModule):
        '''caveat: make sure to provide the MODULE, not the class ### does it really matter?'''
        self._registeredDumpers.append(dumperModule)
    
    def read_db(self):    
        with open(os.path.join(self.basedir, 'dbcore.pickle'), 'r') as f:
            out = pickle.load(f)
            
        for name in out:
            setattr(self, name, out[name])            
            
    def save_db(self):
        '''saves the core data to dbcore.pickle'''
        out = {'_registeredDumpers': self._registeredDumpers} ## things that define the state of this mdb and should be saved
        with open(os.path.join(self.basedir, 'dbcore.pickle'), 'w') as f:
            pickle.dump(out, f)
        
    def itemexists(self, item):
        #todo: this is inefficient, since it has to load all the data
        #just to see if there is a key
        try:
            self.__getitem__(item)
            return True
        except:
            return False
            
    def __getitem__(self, arg):
        try:
            sqllitedict = SqliteDict(os.path.join(self.basedir, 'sqlitedict.db'), autocommit=True)
            dummy = sqllitedict[arg]
            if isinstance(dummy, LoaderWrapper):
                return IO.LoaderDumper.load(os.path.join(self.basedir, dummy.relpath)) 
            else:
                return dummy
        finally:
            sqllitedict.close()
                
    def get_mkdtemp(self, suffix = ''):
        absolute_path = tempfile.mkdtemp(suffix = suffix, dir = self.basedir) 
        relative_path = os.path.relpath(absolute_path, self.basedir)
        return absolute_path, relative_path
        
    def setitem(self, key, item, dumper = None):
        #check if we have writing privilege
        if self.readonly is True:
            raise RuntimeError("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        #this exists, so jupyter notebooks will not crash when they try to write something
        elif self.readonly is 'warning': 
            warnings.warn("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        elif self.readonly is False:
            pass
        else:
            raise RuntimeError("Readonly attribute is in unknown state. Should be True, False or 'warning, but is: %s" % self.readonly)
        
        #find dumper
        if dumper is None:
            for d in self._registeredDumpers:
                if d == 'self' or d.check(item):
                    dumper = d
                    break
    
        assert(dumper is not None)
                
        #if dumper is 'self': store in this DB
        if dumper == 'self':
            try:
                sqllitedict = SqliteDict(os.path.join(self.basedir, 'sqlitedict.db'), autocommit=True)
                sqllitedict[key] = item
            except:
                raise
            finally: 
                sqllitedict.close() 
                
        #if dumper is something else: 
        #generate temp directory, save the object in that directory using dump()
        #wrap the relative path to an LoaderWrapper object and save it to the 
        #internal database
        else:
            basedir_absolute, basedir_relative = self.get_mkdtemp()
            dumper.dump(item, basedir_absolute)
            self.setitem(key, LoaderWrapper(basedir_relative), dumper = 'self')
            
            
        '''
        how can i make the dumper thingy more 
        '''
        
            
    def __setitem__(self, key, item):
        self.setitem(key, item, dumper = None)
                
    def maybe_calculate(self, key, fun, dumper = None):
        try:
            return self[key]
        except:
            with dask.diagnostics.ProgressBar():
                ret = fun()
                self.setitem(key, ret, dumper)
            return ret    
        
    def keys(self):
        try:
            sqllitedict = SqliteDict(os.path.join(self.basedir, 'sqlitedict.db'), autocommit=True)
            keys = sqllitedict.keys()
            return sorted(keys)
        finally:
            sqllitedict.close()             
            

                      
        
    