'''this module contains base settings of the model data base'''

import dask
import pandas as pd
scheduler = dask.multiprocessing.get 
multiprocessing_scheduler = scheduler
show_computation_progress = True
#dask.set_options(get = scheduler)
npartitions = 80