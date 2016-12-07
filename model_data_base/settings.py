import dask
import distributed
scheduler = dask.multiprocessing.get 
multiprocessing_scheduler = scheduler
#dask.set_options(get = scheduler)
npartitions = 80