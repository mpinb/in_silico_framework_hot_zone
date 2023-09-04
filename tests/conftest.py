# Pytest configuration file
# this code will be run before any other pytest code
# even before pytest discovery
# useful to setup whatever needs to be done before the actual testing or test discovery, such as the distributed.client_object_duck_typed
# for setting environment variables, use pytest.ini or .env instead
import os
import pytest
import socket
import Interface as I
from Interface import get_client
from Interface import logger as rootlogger
import logging
log = logging.getLogger(__name__)

suppress_modules_list = ["biophysics_fitting"]

class ModuleFilter(logging.Filter):
    def __init__(self, suppress_modules_list):
        self.suppress_modules_list = suppress_modules_list
    def filter(self, record):
        m = record.getMessage()
        return not any(
            [module_name in m for module_name in self.suppress_modules_list]
            )

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def pytest_configure(config):
    import distributed
    import matplotlib
    import six
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    # Assume dask server and worker are already started
    # These are set up in the github workflow file.
    # If running locally, make sure you have a dask scheduler and dask worker running on these ports

    if six.PY2:
        client = distributed.Client('localhost:28786')
    else:
        client = distributed.Client('localhost:38786')
    log.info("setting distributed duck-typed object as module level attribute")
    distributed.client_object_duck_typed = client
    
    # Setup logging output
    # only log warnings
    rootlogger.setLevel(logging.WARNING)  # set logging level of root logger to WARNING
    # Suppress logs from verbose modules
    for module_name in suppress_logs:
        for handler in rootlogger.handlers:  # should only be one handler: the streamhandler
            handler.addFilter(ModuleFilter(suppress_modules_list))  # suppress logs from this module
        

