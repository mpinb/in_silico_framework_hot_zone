'''
This module deals with API changes in 3rd party modules.
The following 3rd party modules are used: pandas, dask, distributed
'''

import six, yaml, cloudpickle, os, sys

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
        import pandas.compat.pickle_compat #import Unpickler
        with open(file_path, 'rb') as f:
            return pandas.compat.pickle_compat.load(f)

    YamlLoader = yaml.Loader
    import os

    import os

    class Path(object):
        """
        A patch to use basic pathlib.Path functionality in Python 2. This is used in :py:mod:data_base.isf_data_base.isf_data_base
        """
        def __init__(self, *paths):
            self.path = os.path.abspath(os.path.join(*paths))
            self.name = os.path.basename(self.path)

        def __div__(self, other):
            return Path(self.path, other)

        @staticmethod
        def isinstance(obj):
            return isinstance(obj, Path)

        def __getitem__(self, key):
            return self.path[key]

        def __repr__(self):
            return self.path

        def __eq__(self, other):
            return self.path == other.path
        
        def __getattr__(self, attr):
            """
            Delegate attribute access to the underlying string object.
            """
            return getattr(self.path, attr)
        
        def iterdir(self):
            """
            Iterate over the files in this directory. Does not yield any result for the special links '.' and '..'.
            """
            for entry in os.listdir(str(self.path)):
                if entry not in ('.', '..'):
                    yield Path(os.path.join(str(self.path), entry))
        
        def exists(self):
            return os.path.exists(self.path)

        def as_posix(self):
            """
            Return a string representation of the path with forward slashes (/).
            """
            return self.path.replace(os.sep, '/')
        

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
    from pathlib import Path
        

# --------------- compatibility with old versions of ISF (only used by the Oberlaender lab in Bonn)
# For old pickled data. This is to ensure backwards compatibility with the Oberlaender lab in MPINB, Bonn. Last adapted on 25/04/2024
# Since previous versions of this codebase used pickle as a data format, pickle now tries to import modules that don't exist anymore upon loading
# For this reason, we save the renamed packages/modules under an additional name (i.e. their old name)

import simrun
sys.modules['simrun3'] = simrun  # simrun used to be simrun2 and simrun3 (separate packages). Pickle still wants a simrun3 to exist.

def init_mdb_backwards_compatibility():
    """
    Registers model_data_base as a top-level package
    Useful for old pickled data, that tries to import it as a top-level package. model_data_base has since been moved to data_base.model_data_base
    """
    from data_base import model_data_base
    sys.modules['model_data_base'] = model_data_base

def init_data_base_compatibility():
    init_mdb_backwards_compatibility()
