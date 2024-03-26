import pytest, socket, distributed

@pytest.fixture
def client(pytestconfig):
    # Assume dask server and worker are already started
    # These are set up in the github workflow file.
    # If running tests locally, make sure you have a dask scheduler and dask worker running on the ports you want
    hostname = socket.gethostname()
    if "soma" in hostname:
        ip = socket.gethostbyname(hostname).replace('100', '102')
    else:
        ip = 'localhost'
    c = distributed.Client(
        '{}:{}'.format(ip, pytestconfig.getoption("--dask_server_port"))
        )
    def import_mechanisms(): import mechanisms.l5pt as l5pt
    c.run(import_mechanisms)
    return distributed.Client(
        '{}:{}'.format(
            ip,
            pytestconfig.getoption("--dask_server_port")))