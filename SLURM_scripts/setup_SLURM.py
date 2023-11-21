"""The main file to run a complete setup when submitting a SLURM job

Depending on your needs, it sets up the following things on your requested resources:
On process 0:
    1. The locking server
    2. The dask scheduler, and dask workers
    3. The jupyter server

On all other processes:
    1. Reads and respects the locking configuration setup by process 0
    2. Sets up dask workers that communicate with the dask scheduler on process 0
"""

# coding: utf-8

import fasteners
import os
import sys
import asyncio
import time
import configparser
from SLURM_scripts.setup_locking_server import setup_locking_server, setup_locking_config
from SLURM_scripts.setup_dask import setup_dask_scheduler, setup_dask_workers
from SLURM_scripts.setup_jupyter_server import setup_jupyter_server
from contextlib import contextmanager
import argparse
from SLURM_scripts.nbrun import run_notebook
from socket import gethostbyname, gethostname

class StoreDictKeyPair(argparse.Action):
    """https://stackoverflow.com/questions/29986185/python-argparse-dict-arg
    parsing command-line arguments as a dictionary with argparse
    used for notebook run kwargs (not yet implemented in submit.sh or setup_SLURM)

    """

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = eval(v)
        setattr(namespace, self.dest, my_dict)


@contextmanager
def Lock(management_dir):
    # Code to acquire resource, e.g.:
    lock = fasteners.InterProcessLock(os.path.join(management_dir, 'lock'))
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()


def get_process_number(management_dir):
    with Lock(management_dir) as lock:
        p = lock.path  # this is a regular string in Python 2
        if type(p) == bytes:  # Python 3 returns a byte string as lock.path
            p = p.decode("utf-8")
        p += '_sync'
        if not os.path.exists(p):
            with open(p, 'w') as f:
                pass
        with open(p, 'r') as f:
            x = f.read()
            if x == '':
                x = 0
            else:
                x = int(x)
        with open(p, 'w') as f:
            f.write(str(x + 1))
        print('I am process number {}'.format(x))
    return x


def reset_process_number(management_dir):
    with Lock(management_dir) as lock:
        p = lock.path  # this is a regular string in Python 2
        if type(p) == bytes:  # Python 3 returns a byte string as lock.path
            p = p.decode("utf-8")
        p += '_sync'
        with open(p, 'w') as f:
            f.write('')

def read_user_config():
    ### setting up user-defined port numbers ###
    parent_path = os.path.dirname(os.path.dirname(__file__))
    config = configparser.ConfigParser()
    config.read(os.path.join(parent_path, "config", "user_settings.ini"))
    return config

def read_user_port_numbers():
    config = read_user_config()
    ports = config['PORT_NUMBERS']
    # assert port numbers are integers
    ports = {k: int(v) for k, v in ports.items()}
    return ports

def setup(management_dir, launch_jupyter_server=True, wait_for_workers=False):
    """
    Sets up the SLURM job.
    Process 0 will uniquely set up:
        1. The locking server
        2. The dask scheduler
        3. The jupyter server (if launch_jupyter_server is True)
    All processes (including process 0) will set up:
        1. The locking configuration
        2. The dask workers, that will connect to the dask scheduler on process 0

    Args:
        management_dir (str): The directory where the SLURM management files will be stored.
        launch_jupyter_server (bool, optional): Whether to launch a Jupyter server. Defaults to True.
        wait_for_workers (bool|int, optional): Whether to wait for a certain amount of dask workers. Defaults to False (0).
    """
    if not os.path.exists(management_dir):
        try:
            os.makedirs(management_dir)
        except OSError:  # if another process was faster creating it
            pass

    PROCESS_NUMBER = get_process_number(management_dir)
    PORTS = read_user_port_numbers()
    ip = gethostbyname(
        gethostname()
    )  # fetches the ip of the current host, usually "somnalogin01" or "somalogin02"
    os.environ['IP'] = ip
    os.environ['IP_INFINIBAND'] = ip.replace(
        '100', '102')  # a bit hackish, but it works
    if not "IP_MASTER" in os.environ.keys():
        os.environ["IP_MASTER"] = ip
    if not "IP_MASTER_INFINIBAND" in os.environ.keys():
        os.environ["IP_MASTER_INFINIBAND"] = ip.replace('100', '102')

    # Setup for thread 0: launch servers
    if PROCESS_NUMBER == 0:
        setup_locking_server(management_dir, PORTS)
        setup_dask_scheduler(
            management_dir,
            PORTS)  # this process creates scheduler.json and scheduler3.json
        if launch_jupyter_server:
            setup_jupyter_server(management_dir, PORTS)
    
    setup_locking_config(management_dir)
    setup_dask_workers(management_dir, wait_for_workers=wait_for_workers)


def run(
    management_dir,
    launch_jupyter_server=True,
    notebook=None,
    nb_kwargs=None,
    sleep=True,
    wait_for_workers=False):
    """
    Sets up the SLURM job for any job configuration:
        1. A batch job
        2. An interactive job
        3. A notebook job (i.e. a batch job that runs a notebook)
    
    If a notebook is specified, it will be run on process 0.
    If no notebook is specified, this Python process will sleep until SLURM kills it.

    Args:
        management_dir (str): The path to the management directory, where management files will be stored (scheduler3.json, lock_sync, locking_server, lock and jupyter output)
        launch_jupyter_server (bool, optional): Whether to launch a Jupyter server. Defaults to True.
        notebook (str, optional): The path to the notebook to run. Defaults to None.
        nb_kwargs (dict, optional): Additional keyword arguments for running the notebook. Defaults to None.
        sleep (bool, optional): Whether to sleep after setup. Defaults to True.
        wait_for_workers (bool|int, optional): Whether to wait for a certain amount of dask workers. Defaults to False.
    """

    setup(
        management_dir,
        launch_jupyter_server=launch_jupyter_server,
        wait_for_workers=wait_for_workers
        )

    if notebook is not None and PROCESS_NUMBER == 0:
        run_notebook(notebook, nb_kwargs=nb_kwargs)
        exit(0)  # quit SLURM when notebook has finished running
    elif sleep:
        time.sleep(60 * 60 * 24 * 365)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('management_dir')  # non-optional positional argument
    parser.add_argument("--nb_kwargs", dest="nb_kwargs_from_cline", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", nargs='?', const=None)
    # parser.add_argument("--nb_suffix", nargs='?', const="-out", default="-out")
    parser.add_argument("--launch_jupyter_server",
                        default=True,
                        action='store_true')
    parser.add_argument('--notebook_name', nargs='?', const="", default=None)
    args = parser.parse_args()

    MANAGEMENT_DIR = args.management_dir
    LAUNCH_JUPYTER_SERVER = args.launch_jupyter_server  # False by default, if left unspecified

    if LAUNCH_JUPYTER_SERVER:
        print("Launching Jupyter server: {}".format(LAUNCH_JUPYTER_SERVER))
    print('using management dir {}'.format(MANAGEMENT_DIR))

    if args.notebook_name is not None:
        print("Running notebook {}".format(args.notebook_name))
        print("with kwargs {}".format(args.nb_kwargs_from_cline))

    run(
        MANAGEMENT_DIR,
        LAUNCH_JUPYTER_SERVER,
        notebook=args.notebook_name,
        nb_kwargs=args.nb_kwargs_from_cline
    )
