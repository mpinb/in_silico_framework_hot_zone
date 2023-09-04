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
import logging

suppress_logs = ["biophysics_fitting"]

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
    # print("setting distributed duck-typed object as module level attribute")
    distributed.client_object_duck_typed = client
    # only log warnings
    I.logger.setLevel(logging.WARNING)  # set logging level of root logger to WARNING
    for module_name in suppress_logs:
        I.logger.addFilter(logging.Filter(module_name))  # suppress logs from this module
        

