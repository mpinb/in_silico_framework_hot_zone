# Pytest configuration file
# this code will be run before any other pytest code
# even before pytest discovery
# useful to setup whatever needs to be done before the actual testing or test discovery, such as the distributed.client_object_duck_typed
# for setting environment variables, use pytest.ini instead
import os
import pytest
import socket
import Interface
from Interface import get_client
import logging
disable_loggers = ['single_cell_parser']

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
        client = get_client()
    # print("setting distributed duck-typed object as module level attribute")
    distributed.client_object_duck_typed = client
    
    logger_names = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger_to_disable_name in disable_loggers:
        for logger_name in logger_names:
            if logger_to_disable_name in logger_name:
                # disable all loggers if a part of its name appears in disable_loggers
                logging.getLogger(logger_name).disabled = True

        

# other config
