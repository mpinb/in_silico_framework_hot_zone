import dask

try: # new dask versions
    synchronous_scheduler = dask.get
except AttributeError: # old dask versions
    synchronous_scheduler = dask.async.get_sync