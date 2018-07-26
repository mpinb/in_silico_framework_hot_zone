'''
This module deals with API changes in 3rd party modules.
The following 3rd party modules are used: pandas, dask, distributed
'''

import dask

try: # new dask versions
    synchronous_scheduler = dask.get
except AttributeError: # old dask versions
    synchronous_scheduler = dask.async.get_sync
    
multiprocessing_scheduler = dask.multiprocessing.get