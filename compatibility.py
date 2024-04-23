'''
This module deals with API changes in 3rd party modules.
The following 3rd party modules are used: pandas, dask, distributed
'''

import six
import yaml
import cloudpickle
import sys

# try: # new dask versions
#     synchronous_scheduler = dask.get
# except AttributeError: # old dask versions
#     synchronous_scheduler = dask.async.get_sync

# synchronous_scheduler = dask.get

#def mycompute(*args, **kwargs):
#    if six.PY3:
#        if 'get' in kwargs:
#            kwargs['scheduler'] = kwargs['get']
#            del kwargs['get']
#    return dask.compute(*args, **kwargs)

#dask.compute = mycompute


# --------------- compatibility with Python 2.7 vs Python 3.8/3.9
#  multiprocessing_scheduler = dask.multiprocessing.get
from six.moves import cPickle
if six.PY2:

    def pickle_fun(obj, file_path):
        with open(file_path, 'wb') as f:
            cPickle.dump(obj, f)

    def unpickle_fun(file_path):
        with open(file_path, 'rb') as f:
            return cPickle.load(f)

    def cloudpickle_fun(obj, file_path):
        with open(file_path, 'wb') as f:
            cloudpickle.dump(obj, f)

    def uncloudpickle_fun(file_path):
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)

    def pandas_unpickle_fun(file_path):
        # TODO: this is just the cloudpickle thing again. testing for compatibility reasons
        return uncloudpickle_fun(file_path)

    YamlLoader = yaml.Loader

elif six.PY3:
    import types
    types.SliceType = slice

    def pickle_fun(obj, file_path):
        with open(file_path, 'wb') as f:
            cPickle.dump(obj, f, protocol=2)

    def unpickle_fun(file_path):
        with open(file_path, 'rb') as f:
            return cPickle.load(f, encoding='latin1')

    def cloudpickle_fun(obj, file_path):
        with open(file_path, 'wb') as f:
            cloudpickle.dump(obj, f, protocol=2)

    def uncloudpickle_fun(file_path):
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f, encoding='latin1')

    def pandas_unpickle_fun(file_path):
        import pandas.compat.pickle_compat  #import Unpickler
        with open(file_path, 'rb') as f:
            return pandas.compat.pickle_compat.load(f)

    YamlLoader = yaml.FullLoader  # Better choice, but only exists in Py3
    import pandas.core.indexes
    sys.modules['pandas.indexes'] = pandas.core.indexes

# --------------- compatibility with old versions of ISF (only used by the Oberlaender lab in Bonn)
# For old pickled data. This is to ensure backwards compatibility with the Oberlaender lab in MPINB, Bonn. Added on 16/04/2024
# Since previous versions of this codebase used pickle as a data format, pickle now tries to import modules that don't exist anymore upon loading
# For this reason, we save the renamed packages/modules under an additional name (i.e. their old name)
from data_base import model_data_base
import simrun
sys.modules['simrun3'] = simrun  # simrun used to be simrun2 and simrun3 (separate packages). Pickle still wants a simrun3 to exist.
sys.modules['model_data_base'] = model_data_base  # this used to be a top-level package