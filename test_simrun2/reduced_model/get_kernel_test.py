from __future__ import absolute_import
#from ..context import *
from simrun2.reduced_model.get_kernel import _kernel_postprocess
import unittest
import numpy as np

class Tests(unittest.TestCase): 
    def test_kernel_postprocess(self):
        class ClfMock():
                pass    
            
        def generate_test_clfs(n_clfs, n_names, n_lda_values):
            clfs = {}
            l = clfs['classifier_'] = list()
            for lv in range(n_clfs):
                clf = ClfMock()
                l.append(clf)
                clf.coef_ = [np.ones(n_lda_values)]
            return clfs
        
        out = _kernel_postprocess(generate_test_clfs(10, 2, 100))
        self.assertEqual(len(out['kernel_dict'][0]['EXC']), 50)
        self.assertEqual(len(out['kernel_dict'][0]['INH']), 50)
        self.assertEqual(len(out['kernel_dict']), 10)
        
        out = _kernel_postprocess(generate_test_clfs(10, 3, 150), n=3, names = ['a', 'b', 'c'])
        self.assertEqual(len(out['kernel_dict'][0]['a']), 50)
        self.assertEqual(len(out['kernel_dict'][0]['b']), 50)
        self.assertEqual(len(out['kernel_dict'][0]['c']), 50)
        
        clfs = generate_test_clfs(10, 3, 151)
        self.assertRaises(ValueError, lambda: _kernel_postprocess(clfs, n=3, names = ['a', 'b', 'c']))

if __name__ == "__main__":
    testRunner = unittest.TextTestRunner(verbosity = 3)
    suite = unittest.TestLoader().loadTestsFromTestCase(Tests)
    testRunner.run(suite)