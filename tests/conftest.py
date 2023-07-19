# this code will be run before any other pytest code
# even before pytest discovery
# useful to setup whatever needs to be done before the actual testing or test discovery
# for setting environment variables, use pytest.ini instead
import os
import socket


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
    distributed.client_object_duck_typed = client

        

# other config
