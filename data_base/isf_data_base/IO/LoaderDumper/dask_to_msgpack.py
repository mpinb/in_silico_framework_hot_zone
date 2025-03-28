"""
.. deprecated:: 0.2.0
   The pandas-msgpack format is set to be deprecated in the future.
   Please consider using parquet instead.

:skip-doc:
"""


from . import dask_to_categorized_msgpack
import os

Loader = dask_to_categorized_msgpack.Loader
check = dask_to_categorized_msgpack.check


def dump(obj, savedir, repartition=False, scheduler=None, client=None):
    import os
    return dask_to_categorized_msgpack.dump(
        obj,
        savedir,
        repartition=repartition,
        scheduler=scheduler,
        categorize=False,
        client=client
        )
