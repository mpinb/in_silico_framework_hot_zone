import unittest
import test
import model_data_base
import test.decorators

#matplotlib issues: matplotlib.use doesn't set the backend
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#print(matplotlib.get_backend())
from model_data_base import ModelDataBase
plt.switch_backend('agg')  
if __name__ == "__main__":
#     import shutil
#     try:
#         shutil.rmtree('/nas1/Data_arco/model_data_base/test/data/test_temp')
#     except:
#         pass
    ##################################################
    # configure here
    ##################################################
    
    #if testlevel is set to integer, tests with the decorator
    #unittest.monkypatches.run_if_testlevel(level) are only run
    #if testlevel >= level.
    #if testlevel is set to 'all', all tests will run.
    test.decorators.current_testlevel = 1#'all'#0#'all'#'0
#     unittest.monkypatches.testlevel = 'all'
    
    #if run is set to a string resembling the path of a module
    #containing unittest.TestCase object, only the tests in the
    #speified module will run.
    #if run is set to 'all', all tests in the test module will run.
#     run = 'test.tuplecloudsqlitedict_test'
    run = 'all'
    #run = 'test.plotfunctions.manylines_test'
    
    #verbosity of testrunner
    verbosity = 5
    #######################################################
    # end of configuration
    ########################################################
    
    
    testRunner = unittest.TextTestRunner(verbosity = verbosity)
    
    if run == 'all': #testdiscovery
        tests = unittest.defaultTestLoader.discover('.', pattern = '*_test.py')
        testRunner.run(tests)   
    
    else: #run tests in specified module
        test = __import__(run, globals(), locals(), [], 0)
        suite = eval('unittest.TestLoader().loadTestsFromTestCase(%s.Tests)' % run)        
        testRunner.run(suite)