from data_base.isf_data_base.IO.LoaderDumper import (
    # dask_to_parquet,
    dask_to_msgpack,
    dask_to_categorized_msgpack,
    pandas_to_msgpack,
    get_dumper_string_by_dumper_module,
    # pandas_to_parquet,
    to_cloudpickle,
)

DEFAULT_DUMPER = to_cloudpickle
OPTIMIZED_PANDAS_DUMPER = pandas_to_msgpack
OPTIMIZED_DASK_DUMPER = dask_to_msgpack
OPTIMIZED_CATEGORIZED_DASK_DUMPER = dask_to_categorized_msgpack

NEUP_DIR = "parameterfiles_cell_folder"
NETP_DIR = "parameterfiles_network_folder"
HOC_DIR = "morphology"
SYN_DIR = "syn_folder"
CON_DIR = "con_folder"
RECSITES_DIR = "recsites_folder"