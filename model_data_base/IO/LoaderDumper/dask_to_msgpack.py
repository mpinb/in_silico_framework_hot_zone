from . import dask_to_categorized_msgpack
import os

# If this module is called from within the test suite:
TESTING = True if os.environ.get('TESTING') is not None else False

Loader = dask_to_categorized_msgpack.Loader
check = dask_to_categorized_msgpack.check

def dump(obj, savedir, repartition = False, get = None, client = None):
    if not TESTING:
        raise RuntimeError('pandas-msgpack is not supported anymore in the model_data_base') 
    return dask_to_categorized_msgpack.dump(obj, savedir, repartition = repartition, get = get, categorize = False, client = client)
