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
    return c