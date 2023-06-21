
# coding: utf-8

# In[1]:

import fasteners
import os
import shutil
import sys
import yaml
import socket
import time
import configparser

### setting up user-defined port numbers ###
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
config = configparser.ConfigParser()
config.read(os.path.join(__location__, 'user_settings.ini'))
ports = config['PORT_NUMBERS']


#################################
# methods for coordinating jobs
#################################

from contextlib import contextmanager
@contextmanager
def Lock():
    # Code to acquire resource, e.g.:
    lock  = fasteners.InterProcessLock(os.path.join(MANAGEMENT_DIR, 'lock'))    
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()

def get_process_number():
    with Lock() as lock:
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

def reset_process_number():
    with Lock() as lock:
        p = lock.path  # this is a regular string in Python 2
        if type(p) == bytes:  # Python 3 returns a byte string as lock.path
            p = p.decode("utf-8")
        p += '_sync'
        with open(p, 'w') as f:
            f.write('')


#################################################
# setting up locking server
#################################################
def get_locking_file_path():
    return os.path.join(MANAGEMENT_DIR, 'locking_server')

def setup_locking_server():
    print('-'*50)
    print('setting up locking server')
    #command = 'redis-server --save "" --appendonly no --port 8885 --protected-mode no &'    
    #print command
    #os.system(command)
    #config = [dict(type = 'redis', config = dict(host = socket.gethostname(), port = 8885, socket_timeout = 1))]
    config = [{
        'config': {
            'hosts': 'somalogin02-hs:{}'.format(ports['locking_server'])
            }, 
        'type': 'zookeeper'}]
    # config = [{'type': 'file'}]  # uncomment this line if zookeeper is not running (i.e., you receive an error similar to 'No handlers could be found for logger "kazoo.client"')
    with open(get_locking_file_path(), 'w') as f:
        f.write(yaml.dump(config))
    setup_locking_config()
    print('-'*50)

def setup_locking_config():
    print('-'*50)
    print('updating locking configuration to use new server')
    while not os.path.exists(get_locking_file_path()):
        time.sleep(1)
    os.environ['ISF_DISTRIBUTED_LOCK_CONFIG'] = get_locking_file_path() 
    check_locking_config()
    print('-'*50)

def check_locking_config():
    import model_data_base.distributed_lock
    print('locking configuration')
    print(model_data_base.distributed_lock.server)
    print(model_data_base.distributed_lock.client)
    #assert(model_data_base.distributed_lock.server['type'] == 'redis')
    #import socket
    #socket.gethostname()

    #config = [dict(type = 'redis', config = dict(host = socket.gethostname(), port = 8885, socket_timeout = 1))]

    #print 'old locking configuration'
    #print model_data_base.distributed_lock.server
    #print model_data_base.distributed_lock.client

    #import time
    #while True:
    #    try:
    #        model_data_base.distributed_lock.update_config(config)
    #        break
    #    except RuntimeError:
    #        print 'could not connect, retrying in 1 sec'
    #        time.sleep(1) 

    #print 'updated locking configuration'
    #print model_data_base.distributed_lock.server
    #print model_data_base.distributed_lock.client

#################################################
# setting up dask-scheduler
#################################################
def _get_sfile(management_dir):
    return os.path.join(management_dir, 'scheduler.json'), os.path.join(management_dir, 'scheduler3.json')

def setup_dask_scheduler(management_dir):
    print('-'*50)
    print('setting up dask-scheduler')
    sfile, sfile3 = _get_sfile(management_dir)
    command = 'dask-scheduler --scheduler-file={} --port={} --bokeh-port={} --interface=ib0 &'
    command = command.format(sfile, ports['dask_client_2'], ports['dask_dashboard_2'])
    print(command)
    os.system(command)
    command = '''bash -ci "source ~/.bashrc; source_3; dask-scheduler --scheduler-file={} --port={} --interface=ib0 --dashboard-address=:{}" &'''
    command = command.format(sfile3, ports['dask_client_3'], ports['dask_dashboard_3'])
    print(command)
    os.system(command)
    print('-'*50)


#################################################
# setting up dask-worker
#################################################
def setup_dask_workers(management_dir):
    print('-'*50)
    print('setting up dask-workers')
    n_cpus = os.environ['SLURM_CPUS_PER_TASK']
    sfile, sfile3 = _get_sfile(management_dir)
    command = 'dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --memory-limit=100e9 --local-directory $JOB_TMPDIR &'.format(nprocs = n_cpus, sfile = sfile)
    print(command)
    os.system(command)
    command = '''bash -ci "source ~/.bashrc; source_3; dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --local-directory $JOB_TMPDIR --memory-limit=100e9" &'''
    command = command.format(nprocs = n_cpus, sfile = sfile3)
    print(command)
    os.system(command)
    print('-'*50)

##############################################
# setting up jupyter-notebook
#############################################
def setup_jupyter_notebook(management_dir):
    print('-'*50)
    print('setting up jupyter notebook')
    check_locking_config() 
    command = "cd notebooks; jupyter-notebook --ip='*' --no-browser --port={} "
    command = command.format(ports['jupyter_notebook'])
    print(command)
    # Redirect both stdout and stderr (&) to file
    os.system(command + "&>>{} &".format(os.path.join(management_dir,  "jupyter.txt")))    
    print('-'*50)
    #command = "conda activate /axon/scratch/abast/anaconda3/; jupyter-lab --ip='*' --no-browser --port=11113 &"
    #command = 'screen -S jupyterlab -dm bash -c "source ~/.bashrc; source_3; ' +     '''jupyter-lab --ip='*' --no-browser --port=11113"'''
    #command = '''bash -c "source ~/.bashrc; source_3; jupyter-lab --ip='*' --no-browser --port=11113" &'''
    #command = '''(source ~/.bashrc; source_3; jupyter-lab --ip='*' --no-browser --port=11113) &'''
    command = '''bash -ci "source ~/.bashrc; source_3; jupyter-lab --ip='*' --no-browser --port={} --NotebookApp.allow_origin='*' --NotebookApp.ip='0.0.0.0'" '''
    command = command.format(ports['jupyter_lab'])
    #command = "/axon/scratch/abast/anaconda3/bin/jupyter-lab --ip='*' --no-browser --port=11113"
    # append output to same file as notebook (ance the >> operator rather than >)
    os.system(command + "&>>{} &".format(os.path.join(management_dir,  "jupyter.txt")))
# In[8]:
if __name__ == "__main__":
    try:
        MANAGEMENT_DIR = sys.argv[1]
    except IndexError:
        # The python file was run without an argument: likely for testing purposes
        sys.exit(1)
    
    LAUNCH_JUPYTER_SERVER = True  # by default, if left unspecified
    if len(sys.argv) > 2:
        # component_1_SOMA.py was called from submit.sh with extra arguments
        LAUNCH_JUPYTER_SERVER = bool(int(sys.argv[2]))  # only launch when interactive session is started
        print("Launching Jupyter server: {}".format(LAUNCH_JUPYTER_SERVER))
    print('using management dir {}'.format(MANAGEMENT_DIR))


    #if os.path.exists(management_dir):
    #    shutil.rmtree(management_dir)
    if os.path.exists(MANAGEMENT_DIR):
        try:    
            os.makedirs(MANAGEMENT_DIR)
        except OSError: # if another process was faster creating it
            pass
    PROCESS_NUMBER = get_process_number()

    if PROCESS_NUMBER == 0:
        setup_locking_server()
        setup_dask_scheduler(MANAGEMENT_DIR)
        if LAUNCH_JUPYTER_SERVER:
            setup_jupyter_notebook(MANAGEMENT_DIR)
    setup_locking_config()
    setup_dask_workers(MANAGEMENT_DIR)    
    time.sleep(60*60*24*365)
