import dask_to_categorized_msgpack

Loader = dask_to_categorized_msgpack.Loader
check = dask_to_categorized_msgpack.check

def dump(obj, savedir, repartition = False, get = None):  
    return dask_to_categorized_msgpack.dump(obj, savedir, repartition = repartition, get = get, categorize = False)