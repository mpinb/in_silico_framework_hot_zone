from __future__ import absolute_import
from .tuplecloudsqlitedict import SqliteDict
import os, time


###################################################
# additional locking
###################################################


# maybe this locking is over the top and sqlite already handels this well,
# however I sometimes got OperationalError: database is locked exceptions.
# it is now optional to switch on an aditional filebased lock. 
# This has the disadvantage that it also blocks concurrent reading.

locking = True
if locking:
    import fasteners

def aquire_lock(path):
    if not locking: return
    path = path + '.lock'
    mylock = fasteners.InterProcessLock(path)
    mylock.acquire(blocking=True)
    return mylock
        
def release_lock(mylock):
    if not locking: return 
    mylock.release()
    
class SQLiteBackend(object):
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            sqlitedict = self._get_sql() # make sure, sqlite-file exists
            self._close_sql(sqlitedict)
        
    def _get_sql(self):
        self.lock = aquire_lock(self.path)
        return SqliteDict(self.path, autocommit=True)
    
    def _close_sql(self, sqlitedict):
        sqlitedict.close()
        release_lock(self.lock)
    
    def __getitem__(self, arg):
        '''Backend method to retrive item from the database'''
        dict_ = self._vectorized_getitem([arg])
        assert(len(dict_) == 1)
        return dict_.values()[0]
    
    def _vectorized_getitem(self, keys):
        '''this allows to get many values at once, reducing the overhead of repeated 
        opening and closing the connection'''
        try:
            sqllitedict = self._get_sql()    
            dummy = {key: sqllitedict[key] for key in keys}          
        finally:
            self._close_sql(sqllitedict)
        return dummy
    
    def __setitem__(self, key, item):
        '''Backend method to add a key-value pair to the sqlite database'''
        self._vectorized_setitem({key: item})

    def _vectorized_setitem(self, dict_):
        '''this allows to set many values at once, reducing the overhead of repeated 
        opening and closing the connection'''
        try:
            sqllitedict = self._get_sql()
            for k, v in dict_.iteritems():
                sqllitedict[k] = v
        finally: 
            self._close_sql(sqllitedict)    

    def __delitem__(self, arg):
        '''Backend method to delete item from the sqlite database.'''
        self._vectorized_delitem([arg])
            
    def _vectorized_delitem(self, keys):
        '''this allows to delete many values at once, reducing the overhead of repeated 
        opening and closing the connection'''        
        try:
            sqllitedict = self._get_sql()
            for k in keys:
                del sqllitedict[k]          
        finally:
            self._close_sql(sqllitedict)       
    
    def keys(self):
        try:
            sqllitedict = self._get_sql()
            keys = sqllitedict.keys()
            return sorted(keys)
        finally:
            self._close_sql(sqllitedict)