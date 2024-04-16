import pytest, distributed

@pytest.fixture
def client(pytestconfig):
    # Assume dask server and worker are already started
    # These are set up in the github workflow file.
    # If running tests locally, make sure you have a dask scheduler and dask worker running on the ports you want
    c = distributed.Client(
        '{}:{}'.format(
            "localhost", 
            pytestconfig.getoption("--dask_server_port"))
        )
    def import_Interface(): import Interface as I
    c.run(import_Interface)  # to assure all modules are initialized, neuron mechanisms are found, and backwards compatibility is assured.
    return c