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

class ModelDataBase(object):
    def __init__(self, path, tempdir, forceload = False, readonly = False):
        '''
        This is an interface to simulation results. The simulation results located in
        path are converted to an optimized format located in tempdir.
        
        After the initialization, the data can be accessed in the following way:
        mdb['voltage_traces']
        mdb['synapse_activation']
        mdb['spike_times']
        mdb['metadata']
        mdb['cell_activation']
        
        Further more, it is possible to assign new elements to the database
        mdb['my_new_element'] = my_new_element
        These elements are stored together with the other data in the tempdir.
        
        They can be re ad out of the database in the following way:
        my_reloaded_element = mdb['my_new_element']
        
        It is possible to use tuples oif strings as keys.
        To read out all existing keys, use the keys()-function.
        
        After an update of the underlying libraries (e.g. dask), it might be,
        that the standard data ('voltage_traces', 'synapse_activation' and so on) 
        can not be unpickled any more. In this case, you can rebuild these
        Dataframes by calling the _regenerate_data() method. If you regain
        access to the underlying data e.g. mdb['voltage_traces'], you should
        consider to save the updated database with the save_db() method.     
        '''
        
        #todo: rename tempdir to basedir
        
        self.analyze = analyze      
        self._registeredDumpers = [IO.LoaderDumper.to_pickle, 'self']
        self.path = path
        self.tempdir = tempdir
        self.settings = settings #settings is imported above
        self.forceload = forceload
        self.readonly = readonly #possible values: False, True, 'warning'

        if not forceload:       
            self.add_directory(path, tempdir)
        else:
            print('Data integrity check omitted. Data is simply loaded.')
            if not os.path.exists(tempdir): 
                raise RuntimeError("""tempdir %s does not exist. Please specify the path to your simulation results""" % path)
            self.read_db()
            #those information was overwritten by read_db()
            self.path = path
            self.tempdir = tempdir

        #self.old = old.old(self)
        
        
    def registerDumper(self, dumperModule):
        '''caveat: make sure to procide the MODULE, not the class'''
        self._registeredDumpers.append(dumperModule)
            
    def add_directory(self, path,tempdir):
        #todo: this should to be put in the Converrter class
        #currently, path and tempdir do not have any impact, because
        #because the called functions only look on self.path and self.tempdir
        
        #what should this function implement based on its name?
        #it should provide an interface, to add arbitrary simulation
        #folders to the mdb.
        
        #how could this be done better
        #the model_data_base class only implements the set_item method
        #this method can than be called by a Converter object or function,
        #which either sets the data directly, or sets an respective Loader instance.
        #For each such object, the Converter object is pickled in the metadata,
        #so in case of corrupted data, the data necessary for the Loader can be regenerated.
        
        #this implies, that the set_item method allways sets a dictionary, containing
        #two fields: data (for the actual object to be stored) and metadata
        
        #on initialization of the model_data_base, an optional selfcheck can be made
        #in which the .check() method on every loader object is called.
        if not os.path.exists(path): 
            raise RuntimeError("""path %s does not exist. Please specify the path to your simulation results""" % path)
        if self.check_already_build():
            self.read_db()
        else:
            self.build_db() #create them
            self.save_db()
    
    def check_already_build(self):
        '''checks if SOME build is in the folder'''
        #no folder, no files --> suitable for rebuild
        if not os.path.exists(self.tempdir): return False
        #folder exists --> not suitable for rebuild
        else:
            db_main_file = os.path.join(self.tempdir, 'dbcore.pickle')
            #folder exists, but not the necessary file
            if not os.path.exists(db_main_file): 
                print(db_main_file)
                raise RuntimeError("The specified tempdir exists, but it does not contain"
                                   + " the necessary dbcore.pickle. If you want to rebuild the database,"
                                   + " please provide a directory, that currently does not exist." 
                                   + " The directory will be created and the database will be" 
                                   + " rebuild in this folder.")
            #file exists
            else:
                try:
                    with open(db_main_file, 'r') as f:
                        pickle.load(f)
                    #file exists and can be read
                    return True
                except:
                    raise RuntimeError("The speicied tempdir exists. It contains the necessary dbcore.pickle"
                                    + "However, this file can not be unpickled. You might consider rebuilding"
                                    + "the database by providing a tempdir, that currently does not exist." 
                                    + " The directory will be created and the database will be" 
                                    + " rebuild in this folder.")           
    def build_db(self):    
        self._build_db_part1()
        self._build_db_part2()
        
    def _build_db_part1(self):
        '''builds the metadata object and rewrites files for fast access.
        Only needs to be called once to put the necessary files in the tempdir'''
        print('building database ...')
        #make filelist of all soma-voltagetraces-files
        self.file_list = IO.make_file_list(self.path, 'vm_all_traces.csv')
        print('done with filelist ...')        
        #read all soma voltage traces in dask dataframe
        self.voltage_traces = IO.read_voltage_traces_by_filenames(self.path, self.file_list)
        print('done with voltage_traces ...')        
        #the indexes of this dataframe are stored for further use to identify the 
        #simulation trail
        self.sim_trails = self.voltage_traces.index.compute()
        print('unambiguous sim_trail_indices generated ...')        
        #builds the metadata object, which connects the sim_trail indexes with the 
        #associated files
        self.metadata = IO.create_metadata(self.sim_trails)
        print('finished generating metadata ...')        
        #rewrites the synapse and cell files in a way they can be acessed fast
        print('start rewriting synapse and cell activation data in optimized format')                
        IO.rewrite_data_in_fast_format(self)    
        print('data is written. The above steps will not be necessary again if the' \
              + 'ModelDataBase object is instantiated in the same way.')                
        
    
    def _build_db_part2(self):
        self.spike_times = analyze.spike_detection(self.voltage_traces)
        self.synapse_activation = IO.read_synapse_activation_times(self)
        self.cell_activation = IO.read_cell_activation_times(self)
        
    def _regenerate_data(self):
        self.voltage_traces = IO.read_voltage_traces_by_filenames(self.path, self.file_list)
        self._build_db_part2()
    
    def read_db(self, selfcheck = True):
            ###path is already checked
#         if not os.path.exists(self.path): 
#             raise RuntimeError("There is something wrong with the tempdir %s." +
#                                "The folder exists, but the file dbcore.pickle could\n" + 
#                                "not be found. Plese delete the folder manually \n" +
#                                "(so it can be rebuild) or use another tempdir.")
        
        with open(os.path.join(self.tempdir, 'dbcore.pickle'), 'r') as f:
            out = pickle.load(f)
        
        #check accordance of saved and provided data location
        if not self.forceload:
            if not os.path.abspath(self.tempdir) == os.path.abspath(out['tempdir']):
                raise RuntimeError("Cave: the tempdir specified in the saved data \n" + 
                                     "differs from the real location.\n"+
                                     "given location:%s \n" % self.tempdir+
                                     "location stored in file:%s \n" % out['tempdir'])
            if not os.path.abspath(self.path) == os.path.abspath(out['path']):
                raise RuntimeError("Cave: the path specified in the saved data " + 
                                     "differs from the real location \n" + 
                                    "given location:%s \n" % self.path +
                                     "location stored in file:%s \n" % out['path'])
            
        for name in out:
            setattr(self, name, out[name])            
        
        if selfcheck:
            pass
        
        #self._build_db_part2()
    
    def save_db(self):
        '''saves the core data to dbcore.pickle'''
        out = {'metadata': self.metadata, 'path': self.path, 
               'tempdir': self.tempdir, 'file_list': self.file_list,
               'sim_trails': self.sim_trails, 'voltage_traces': self.voltage_traces,
               'spike_times': self.spike_times,
               'synapse_activation': self.synapse_activation,
               'cell_activation': self.cell_activation}
        with open(os.path.join(self.tempdir, 'dbcore.pickle'), 'w') as f:
            pickle.dump(out, f)
        
    def itemexists(self, item):
        try:
            self.__getitem__(item)
            return True
        except:
            return False
            
    def __getitem__(self, arg):
        #everything should go in self.sqllitedict
        if arg == 'spike_times':
            return deepcopy(self.spike_times)
        elif arg == 'voltage_traces':
            return deepcopy(self.voltage_traces)
        elif arg == 'synapse_activation':
            return deepcopy(self.synapse_activation)
        elif arg == 'cell_activation':
            return deepcopy(self.cell_activation)
        elif arg == 'metadata':
            return deepcopy(self.metadata)
        else:
            try:
                sqllitedict = SqliteDict(os.path.join(self.tempdir, 'sqlitedict.db'), autocommit=True)
                dummy = sqllitedict[arg]
                if isinstance(dummy, LoaderWrapper):
                    return IO.LoaderDumper.load(os.path.join(self.tempdir, dummy.relpath)) 
                else:
                    return dummy
            finally:
                sqllitedict.close()
                
    def get_mkdtemp(self, suffix = ''):
        absolute_path = tempfile.mkdtemp(suffix = suffix, dir = self.tempdir) 
        relative_path = os.path.relpath(absolute_path, self.tempdir)
        return absolute_path, relative_path
        
    def setitem(self, key, item, dumper = None):
        #check if we have writing privilege
        if self.readonly is True:
            raise RuntimeError("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        elif self.readonly is 'warning':
            warnings.warn("DB is in readonly mode. Blocked writing attempt to key %s" % key)
        elif self.readonly is False:
            pass
        else:
            raise RuntimeError("Readonly attribute is in unknown state. Should be True, False or 'warning, but is: %s" % self.readonly)
        
        #find dumper
        if dumper is None:
            for d in self._registeredDumpers:
                if d.check(item):
                    dumper = d
                    break
    
        assert(dumper is not None)
                
        #if dumper is 'self': store in this DB
        if dumper == 'self':
            try:
                sqllitedict = SqliteDict(os.path.join(self.tempdir, 'sqlitedict.db'), autocommit=True)
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
            tempdir_absolute, tempdir_relative = self.get_mkdtemp()
            dumper.dump(item, tempdir_absolute)
            self.setitem(key, LoaderWrapper(tempdir_relative), dumper = 'self')
        
            
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
            sqllitedict = SqliteDict(os.path.join(self.tempdir, 'sqlitedict.db'), autocommit=True)
            keys = sqllitedict.keys() + ['spike_times', 'voltage_traces', 'synapse_activation', 'cell_activation', 'metadata']
            return sorted(keys)
        finally:
            sqllitedict.close()                       
        
    