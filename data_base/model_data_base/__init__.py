# For backwards compatibility, register model_data_base as top-level module, 
# so pickled data still knows where to find modules
import sys
sys.modules['model_data_base'] = sys.modules[__name__]

from .model_data_base import ModelDataBase
import logging

logger = logging.getLogger("ISF").getChild(__name__)
