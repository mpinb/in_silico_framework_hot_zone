
# coding: utf-8

# In[14]:

import model_data_base
import dask

import os
import shutil
import dask
def maybe_make_dirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def maybe_rmtree(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
from model_data_base.model_data_base import ModelDataBase as MDB
#x = MDB('/abast/delete_me/nernst_mdbv2/')
x = MDB('/abast/delete_me/crossing-over_constant_time_265ms_mdbv2')


# In[15]:

import model_data_base.IO.LoaderDumper.to_pickle
x._registeredDumpers = [model_data_base.IO.LoaderDumper.to_pickle]


# In[16]:

x.keys()


# In[17]:

from model_data_base import tuplecloudsqlitedict
#sqllitedict = tuplecloudsqlitedict.SqliteDict(os.path.join('/abast/delete_me/nernst/', 'sqlitedict.db'), autocommit=True)
sqllitedict = tuplecloudsqlitedict.SqliteDict(os.path.join('/abast/delete_me/crossing-over_constant_time_265ms', 'sqlitedict.db'), autocommit=True)


# In[18]:

sqllitedict.keys()


# In[20]:

for key in sqllitedict.keys():
    print key
    x[key] = sqllitedict[key]


# In[21]:

x[('backup', 'metadata')].head()


# In[ ]:

x['synapse_activation'] = x[('backup', 'synapse_activation')]
x['cell_activation'] = x[('backup', 'cell_activation')]
x['voltage_traces'] = x[('backup', 'voltage_traces')]


# In[22]:

x['metadata'] = x[('backup', 'metadata')]


# In[24]:

del x[('backup', 'synapse_activation')]
del x[('backup', 'cell_activation')]
del x[('backup', 'voltage_traces')]
del x[('backup', 'metadata')]


# In[25]:

x.keys()


# In[ ]:



