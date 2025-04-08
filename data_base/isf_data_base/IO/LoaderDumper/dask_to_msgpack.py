"""Save and load dask dataframes to msgpack.

This uses a fork of the original `pandas_to_msgpack` package, `available on PyPI <https://pypi.org/project/isf-pandas-msgpack/>`_

See also:
    :py:mod:`~data_base.isf_data_base.IO.LoaderDumper.dask_to_parquet` for saving dask dataframes to parquet files.

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
