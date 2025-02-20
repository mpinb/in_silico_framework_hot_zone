from tests import client as test_dask_client
import pytest

@pytest.fixture(scope="session")
def client(pytestconfig):
    port = pytestconfig.getoption("--dask_server_port")
    c = test_dask_client(port)
    return c
