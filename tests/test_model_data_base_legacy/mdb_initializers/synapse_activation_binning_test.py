import tempfile
import warnings
from tests.test_model_data_base_legacy import *
from model_data_base.mdb_initializers.load_simrun_general \
            import optimize as optimize_simrun_general
from model_data_base.mdb_initializers.synapse_activation_binning \
            import init as init_synapse_activation

from model_data_base.IO.LoaderDumper import dask_to_csv, dask_to_msgpack, dask_to_categorized_msgpack
from model_data_base.utils import silence_stdout
import distributed

optimize_simrun_general = silence_stdout(optimize_simrun_general)


def test_API(fresh_mdb_legacy, client):
    optimize_simrun_general(fresh_mdb_legacy, client=client)
    init_synapse_activation(fresh_mdb_legacy, groupby='EI')
    init_synapse_activation(fresh_mdb_legacy, groupby=['EI'])
    init_synapse_activation(fresh_mdb_legacy, groupby=['EI', 'proximal'])
