import pytest, distributed
from distributed.diagnostics.plugin import SchedulerPlugin

def import_worker_requirements():
    import compatibility
    import mechanisms

class SetupWorker(SchedulerPlugin):
    def __init__(self):
        pass

    def add_worker(scheduler, worker):
        """
        This gets called every time a new worker is added to the scheduler
        """
        scheduler.submit(import_worker_requirements, workers=[worker])
        import_worker_requirements()

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
    c.register_scheduler_plugin(SetupWorker)
    c.run(import_worker_requirements) 
    return c
