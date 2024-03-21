from __future__ import absolute_import
import os
import sys
from tests.context import TEST_DATA_FOLDER

parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, parent)

assert os.path.exists(TEST_DATA_FOLDER)