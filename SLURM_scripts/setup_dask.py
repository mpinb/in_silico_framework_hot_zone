import os
from SLURM_scripts.utils import get_user_port_numbers
from socket import gethostbyname, gethostname
from dask.distributed import Client
import six

#################################################
# setting up dask-scheduler
#################################################
def _get_sfile(management_dir):
    return os.path.join(management_dir,
                        'scheduler.json'), os.path.join(management_dir,
                                                        'scheduler3.json')


def setup_dask_scheduler(management_dir, ports):
    """Set up dask scheduler
    This process is normally exectuted by only one thread on the cluster.
    It keeps track of all the dask workers, distributes tasks and fetches results.

    Args:
        management_dir (str): location of the management dir
        ports (dict | dict-like): A dictionary of port numbers to use for the dask setup.
            Must containg the following keys: 'dask_client_2', 'dask_dashboard_2', 'dask_client_3' and 'dask_dashboard_3'
            Each key must have a port number as value.
            Should be specified in config/user_settings.ini
    """
    from distributed.versions import get_versions
    print("versions:\n", get_versions())
    print('-' * 50)
    print('setting up dask-scheduler')
    sfile, sfile3 = _get_sfile(management_dir)
    # command = 'dask-scheduler --scheduler-file={} --port={} --bokeh-port={} --interface=ib0 &'
    # command = command.format(sfile, ports['dask_client_2'],
    #                          ports['dask_dashboard_2'])
    # print(command)
    # os.system(command)
    command = '''dask-scheduler --scheduler-file={} --port={} --interface=ib0 --dashboard-address=:{} &'''
    command = command.format(sfile3, ports['dask_client_3'],
                             ports['dask_dashboard_3'])
    print(command)
    os.system(command)
    print('-' * 50)


#################################################
# setting up dask-worker
#################################################
def setup_dask_workers(management_dir, wait_for_workers=False):
    """Set up dask workers.
    This process is done by all threads normally (even process 0).
    It sets up dask workers which receive tasks, compute them, and send the result to the dask scheduler.

    Args:
        management_dir (str): The location of the management dir
    """
    print('-' * 50)
    print('setting up dask-workers')
    n_cpus = os.environ['SLURM_CPUS_PER_TASK']
    sfile, sfile3 = _get_sfile(management_dir)
    # command = 'dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --memory-limit=100e9 --local-directory $JOB_TMPDIR &'.format(
    #     nprocs=n_cpus, sfile=sfile)
    # print(command)
    # os.system(command)
    command = '''dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --local-directory $TMPDIR --memory-limit=100e9 &'''
    command = command.format(nprocs=n_cpus, sfile=sfile3)
    print(command)
    os.system(command)
    print('-' * 50)
    if wait_for_workers:
        get_client().wait_for_workers(n_workers=1)


def get_client(timeout=120):
    """Gets the distributed.client object if dask has been setup

    Returns:
        Client: the client object
    """
    ports = get_user_port_numbers()
    if six.PY2:
        client_port = ports['dask_client_2']
    else:
        client_port = ports['dask_client_3']

    if "IP_MASTER" in os.environ.keys():
        if "IP_MASTER_INFINIBAND" in os.environ.keys():
            ip = os.environ['IP_MASTER_INFINIBAND']
        else:
            ip = os.environ["IP_MASTER"]
    else:
        hostname = gethostname()
        ip = gethostbyname(
            hostname
        )  # fetches the ip of the current host, usually "somnalogin01" or "somalogin02"
        if 'soma' in hostname:
            #we're on the soma cluster and have infiniband
            ip = ip.replace('100', '102')  # a bit hackish, but it works
    print("getting client with ip {}".format(ip))
    c = Client(ip + ':' + client_port, timeout=timeout)
    print("got client {}".format(c))
    return c
