"""
Efficient, scalable, and flexible database whose API mimics a python dictionary. 

this package provides efficient and scalable methods to store and access simulation results at a terrabyte scale. 
It also generates metadata, indicating when the data was put in the database and the exact version of the source code that was used at this timepoint. 
Simulation results from the :py:mod:`single_cell_parser` and :py:mod:`simrun` modules can be imported and converted to a high performance binary format. 
Afterwards the data is accessible using the pandas data analysis library and dask.

This package has dynamic ``data_base``, ``IO`` and ``db_initializers`` subpackages. Which data base system to use is decided by :py:mod:`~data_base.data_base`.
This way:
1. future changes to file formats can be easily implemented by creating a new IO subpackage and having it register under the name data_base.IO.
2. Only one IO subpackage is active at a time.
3. Backwards compatibility to old data or old formats is guaranteed, as the previous IO subpackages are still available.

See also:
    :py:mod:`data_base.data_base` to see how old data formats are automatically read in.

All methods in ISF should always use the ``data_base.IO`` submodule, and never import readers or writers directly from a data base system.

Available data base systems:
1. :py:mod:`~data_base.isf_data_base`
2. :py:mod:`~data_base.model_data_base` (deprecated)
"""
import compatibility
compatibility.init_data_base_compatibility()
from .data_base import DataBase
