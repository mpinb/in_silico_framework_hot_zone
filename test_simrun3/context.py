from __future__ import absolute_import
import os
import sys
import tempfile
import getting_started

parent = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(os.path.dirname(__file__), 'data')

sys.path.insert(0, parent)