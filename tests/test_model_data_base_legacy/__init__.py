"""
Since commit 30e4b69239c46083702bd1da15c9bf6254c51a97 model_data_base has been updated, and the old model_data_base has been renamed to model_data_base_legacy. 
This test suite is supposed to test functionality of the old model_data_base.

Th enew model_data_base is still capable of reading in data that waas saved with the old model_data_base. You will see that the LoaderDumper module checks
for the existence of Loader.pickle (the old format) and reads it in accordingly.

The only reason why one would still want to use the legacy mdb version is if:
- you need to _write_ in the old format
- you need to use mdb in Python2.7 (the new mdb is Py3.4< only)
"""

from __future__ import absolute_import
import os, sys, shutil, tempfile
import distributed
import pytest

parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent)
import distributed
from model_data_base import *
import getting_started
import mechanisms
# set up paths
test_data_folder = os.path.join(getting_started.parent, \
                              'example_simulation_data', \
                              'C2_evoked_UpState_INH_PW_1.0_SuW_0.5_C2center/')

