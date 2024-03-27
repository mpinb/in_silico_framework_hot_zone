import pytest, socket, distributed

@pytest.fixture
def client(pytestconfig):
    # Assume dask server and worker are already started
    # These are set up in the github workflow file.
    # If running tests locally, make sure you have a dask scheduler and dask worker running on the ports you want
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    if "soma" in hostname and not hostname[-3] == "2":
        ip = ip.replace('100', '102')
    c = distributed.Client(
        '{}:{}'.format(
            ip, 
            pytestconfig.getoption("--dask_server_port"))
        )
    def import_mechanisms(): import mechanisms.l5pt as l5pt
    c.run(import_mechanisms)
    return c