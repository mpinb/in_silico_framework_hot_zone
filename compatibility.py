'''
This module deals with API changes in 3rd party modules.
The following 3rd party modules are used: pandas, dask, distributed
'''

import dask
import six

# try: # new dask versions
#     synchronous_scheduler = dask.get
# except AttributeError: # old dask versions
#     synchronous_scheduler = dask.async.get_sync

synchronous_scheduler = dask.get

#def mycompute(*args, **kwargs):
#    if six.PY3:
#        if 'get' in kwargs:
#            kwargs['scheduler'] = kwargs['get']
#            del kwargs['get']
#    return dask.compute(*args, **kwargs)

#dask.compute = mycompute
    
#  multiprocessing_scheduler = dask.multiprocessing.get