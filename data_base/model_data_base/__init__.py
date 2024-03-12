# For backwards compatibility, register model_data_base as top-level module, 
# so pickled data still knows where to find modules
import sys
# register model_data_base subpackages and submodules under data_base.module_data_base
sys.modules['model_data_base'] = sys.modules[__name__]

from .model_data_base import ModelDataBase