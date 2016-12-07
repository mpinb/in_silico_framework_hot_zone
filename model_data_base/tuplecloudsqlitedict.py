''''
Basically the same as cloudsqllitedict, however, it supports tuples as keys. This comes at the cost,
that '@' in keys is not allowed anymore.

The class ToupleCloudSqlitedict does not inherit from cloudsqlitedict, however it contains a cloudsqlitedict.
In case, some API is missing, simply extend this class accordingly.
'''

import cloudsqlitedict

class SqliteDict:
    def __init__(self, basedir, autocommit = False):
        self.sqlitedict = cloudsqlitedict.SqliteDict(basedir, autocommit = autocommit)
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            key = '@'.join(key)
        self.sqlitedict.__setitem__(key, value)
        
    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = '@'.join(key)
        return self.sqlitedict.__getitem__(key)
    
    def __delitem__(self, key):
        if isinstance(key, tuple):
            key = '@'.join(key)    
        return self.sqlitedict.__delitem__(key)
    
    def keys(self):
        list_ = self.sqlitedict.keys()
        out = []
        for l in list_:
            if '@' in l:
                out.append(tuple(l.split('@')))
            else:
                out.append(l)
        return out

    def close(self):
        self.sqlitedict.close()
