'''
Created on Aug 15, 2016

@author: arco
'''

import os
import shutil
import tempfile
import cloudpickle as pickle
import IO, IO.LoaderDumper.to_pickle
import IO.LoaderDumper.pandas_to_msgpack
import analyze
import dask.diagnostics
import settings
from tuplecloudsqlitedict import SqliteDict
from copy import deepcopy
import warnings
import re


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
    '''wrapper class pointing to loader, which can be saved in the internal SQL database'''
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
                               + "or write permission is missing. The specified path is %s" % working_dir)
    else:
        try: 
            os.makedirs(working_dir) #todo: should ideally be mkdirs
            return
        except OSError:
            raise MdbException("Can't build database: " \
                               + "Cannot create the directories specified in %s" % working_dir)

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
        
        self.basedir = os.path.abspath(basedir)
        #self.basedir = basedir
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
    
    def __direct_dbget(self, arg):
        try:
            sqllitedict = SqliteDict(os.path.join(self.basedir, 'sqlitedict.db'), autocommit=True)
            dummy = sqllitedict[arg]
        finally:
            sqllitedict.close() 
        return dummy
    
    def __direct_dbset(self, key, item):
        try:
            sqllitedict = SqliteDict(os.path.join(self.basedir, 'sqlitedict.db'), autocommit=True)
            sqllitedict[key] = item
        except:
            raise
        finally: 
            sqllitedict.close()  

    def __direct_dbdel(self, arg):
        try:
            sqllitedict = SqliteDict(os.path.join(self.basedir, 'sqlitedict.db'), autocommit=True)
            del sqllitedict[arg]
        finally:
            sqllitedict.close() 
                           
    def __getitem__(self, arg):
        dummy = self.__direct_dbget(arg)
        if isinstance(dummy, LoaderWrapper):
            return IO.LoaderDumper.load(os.path.join(self.basedir, dummy.relpath)) 
        else:
            return dummy
                
    def get_mkdtemp(self, prefix = '', suffix = ''):
        absolute_path = tempfile.mkdtemp(prefix = prefix + '_', suffix = '_' + suffix, dir = self.basedir) 
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
        
        #check if there is already a subdirectory assigned to this key. If so: delete subdirectory.
        if key in self.keys():
            self.__delitem__(key)
                
        #find dumper
        if dumper is None:
            for d in self._registeredDumpers:
                if d == 'self' or d.check(item):
                    dumper = d
                    break
    
        assert(dumper is not None)
                
        #if dumper is 'self': store in this DB
        if dumper == 'self':
            self.__direct_dbset(key, item)
                
        #if dumper is something else: 
        #generate temp directory, save the object in that directory using dump()
        #wrap the relative path to an LoaderWrapper object and save it to the 
        #internal database
        else:
            basedir_absolute, basedir_relative = self.get_mkdtemp(prefix = slugify(key))
            try:
                dumper.dump(item, basedir_absolute)
                self.__direct_dbset(key, LoaderWrapper(basedir_relative))
            except:
                shutil.rmtree(basedir_absolute)
                if key in self.keys():
                    del self[key]
                raise
               
    def __setitem__(self, key, item):
        self.setitem(key, item, dumper = None)

    def __delitem__(self, key):
        dummy = self.__direct_dbget(key)
        if isinstance(dummy, LoaderWrapper):
            try:
                shutil.rmtree(os.path.join(self.basedir,dummy.relpath))
            except OSError:
                print('The folder ' + os.path.join(self.basedir,dummy.relpath) + ' was registered as belonging to ' + \
                      str(key) + '. I tried to delete this folder, because the corresponding key was overwritten. ' + \
                      'Could not delete anything, because folder did not exist in the first place. I just carry on ...')
        
        self.__direct_dbdel(key)
                       
                
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
            

                      
        
    