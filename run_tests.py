import unittest

# switch matplotlib backend to make sure that test suite can 
# runs on machines that do not have graphic libraries installed (Qt, ...)
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg') 

# import modules, whose tests should run 
import test_model_data_base, test_model_data_base.decorators
import test_simrun2, test_simrun2.decorators
import test_single_cell_parser, test_single_cell_parser.decorators
import distributed


#######################################
# SET TESTLEVEL HERE
# Choose a low testlevel if you only want to run quick tests.
#
# The testlevel of tests can be set using a decorator defined in:
# @decorators.testlevel(some_number)
#
# Valid values are integers and 'all'
####################################### 
test_model_data_base.decorators.current_testlevel = 10#'all'#0#'all'#'0
test_simrun2.decorators.current_testlevel = 10#'all'#0#'all'#'0

#######################################git git
# SELECT TESTS YOU WANT TO RUN HERE
#
# Valid values are strings, which contain a module name, that can be 
# imported. Choose '.' to run all available tests within this module
#######################################
run = '.'#test_single_cell_parser.reader_test'
#run = 'test_model_data_base.model_data_base_test'
#run = 'test_model_data_base.mdb_initializers.load_simrun_general_test'
#run = 'test_simrun2.simrun_test'
#run = 'test_model_data_base.sqlite_backend.sqlite_backend_test'
#run = 'test_model_data_base.model_data_base_register_test'
################################
# verbosity of testrunner
################################
verbosity = 2
    
    
if __name__ == "__main__":
    testRunner = unittest.TextTestRunner(verbosity = verbosity)

if run == '.':
    tests = unittest.defaultTestLoader.discover(run, pattern = '*_test.py')
    testRunner.run(tests)          
else:
    test = __import__(run, globals(), locals(), [], 0)
    suite = eval('unittest.TestLoader().loadTestsFromTestCase(%s.Tests)' % run)        
    testRunner.run(suite)