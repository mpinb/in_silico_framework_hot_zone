from ..context import *
from  model_data_base.IO.voltagetraces import *

import os
import unittest
import glob
class Tests(unittest.TestCase):
    def test_size(self):
        prefix = os.path.join(parent, 'test/data/test_data')
        fnames = os.path.join(parent, 'test/data/test_data', '*', '*', '*_vm_all_traces.csv')
        fnames = glob.glob(str(fnames))
        result = read_voltage_traces(prefix, [fnames[0]])#
        self.assertEqual(result.compute().shape, (10, 13802))
        
if __name__ == "__main__":
    unittest.main()