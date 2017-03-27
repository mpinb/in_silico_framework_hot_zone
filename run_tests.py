import os, shutil
import unittest
import test_model_data_base, test_model_data_base.decorators, test_model_data_base.context
#import test_simrun2, test_simrun2.decorators
import model_data_base
import simrun2


#####################################################################
# make sure everything is in defined state before running tests.
##########################################################
# todo: maybe put these setup routines somewhere else?

# set up mdb
test_mdb_folder = test_model_data_base.context.test_mdb_folder
if os.path.exists(test_mdb_folder):
    shutil.rmtree(test_mdb_folder)
mdb = model_data_base.ModelDataBase(test_mdb_folder)
mdb.settings.show_computation_progress = False
model_data_base.mdb_initializers.load_simrun_general.init(mdb, \
                                  test_model_data_base.context.test_data_folder) 

# setup folder for output files by test suite
files_generated_by_tests =  test_model_data_base.context.files_generated_by_tests
if os.path.exists(files_generated_by_tests):
    shutil.rmtree(files_generated_by_tests)
os.makedirs(files_generated_by_tests)


########################################################
# end of setup
########################################################


# switch matplotlib backend to make sure that test suite can 
# runs on machines that do not have graphic libraries installed (Qt, ...)

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

 
if __name__ == "__main__":

    ##################################################
    # configure test runs here
    ##################################################
    
    #if testlevel is set to integer, tests with the decorator
    #unittest.monkypatches.run_if_testlevel(level) are only run
    #if testlevel >= level.
    #if testlevel is set to 'all', all tests will run.
    test_model_data_base.decorators.current_testlevel = 1#'all'#0#'all'#'0
#    test_simrun2.decorators.current_testlevel = 1#'all'#0#'all'#'0
    
    
#     unittest.monkypatches.testlevel = 'all'
    
    #if run is set to a string resembling the path of a module
    #containing unittest.TestCase object, only the tests in the
    #speified module will run.
    #if run is set to 'all', all tests in the test module will run.
#    run = 'test_model_data_base.model_data_base_test'
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