
# coding: utf-8

# In[1]:

import fasteners
import os
import shutil
import sys
import yaml
import socket
import time
# In[2]:

management_dir = '/ptmp/abast/management_dir_test'
management_dir = sys.argv[1]
print 'using management dir' , management_dir

# In[3]:

#if os.path.exists(management_dir):
#    shutil.rmtree(management_dir)
if not os.path.exists(management_dir):
    try:    
        os.makedirs(management_dir)
    except OSError: # if another process was faster creating it
        pass

# In[21]:

#################################
# methods for coordinating jobs
#################################

from contextlib import contextmanager
@contextmanager
def Lock():
    # Code to acquire resource, e.g.:
    lock  = fasteners.InterProcessLock(os.path.join(management_dir, 'lock'))    
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()

def get_process_number():
    with Lock() as lock:
        p = lock.path+'_sync'
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
        print 'I am process number {}'.format(x)
    return x

def reset_process_number():
    with Lock() as lock:
        with open(lock.path+'_sync', 'w') as f:
            f.write('')

process_number = get_process_number()


# In[5]:

#################################################
# setting up locking server
#################################################
def get_locking_file_path():
    return os.path.join(management_dir, 'locking_server')

def setup_locking_server():
    print '-'*50
    print 'setting up locking server'
    command = 'screen -S redis_server_test -dm bash -c "source ~/.bashrc; source_isf; redis-server --save "" --appendonly no --port 8885 --protected-mode no"'
    command = 'redis-server --save "" --appendonly no --port 8885 --protected-mode no &'    
    print command
    os.system(command)
    config = [dict(type = 'redis', config = dict(host = socket.gethostname(), port = 8885, socket_timeout = 1))]
    with open(get_locking_file_path(), 'w') as f:
        f.write(yaml.dump(config))
    setup_locking_config()
    print '-'*50

def setup_locking_config():
    print '-'*50
    print 'updating locking configuration to use new server'
    while not os.path.exists(get_locking_file_path()):
        time.sleep(1)
    os.environ['ISF_DISTRIBUTED_LOCK_CONFIG'] = get_locking_file_path() 
    check_locking_config()
    print '-'*50

def check_locking_config():
    import model_data_base.distributed_lock
    print 'locking configuration'
    print model_data_base.distributed_lock.server
    print model_data_base.distributed_lock.client
    assert(model_data_base.distributed_lock.server['type'] == 'redis')
#    import socket
#    socket.gethostname()
#
#    config = [dict(type = 'redis', config = dict(host = socket.gethostname(), port = 8885, socket_timeout = 1))]
#
#    print 'old locking configuration'
#    print model_data_base.distributed_lock.server
#    print model_data_base.distributed_lock.client
#
#    import time
#    while True:
#        try:
#            model_data_base.distributed_lock.update_config(config)
#            break
#        except RuntimeError:
#            print 'could not connect, retrying in 1 sec'
#            time.sleep(1) 
#
#    print 'updated locking configuration'
#    print model_data_base.distributed_lock.server
#    print model_data_base.distributed_lock.client


# In[6]:

#################################################
# setting up dask-scheduler
#################################################
def _get_sfile(management_dir):
    return os.path.join(management_dir, 'scheduler.json')

def setup_dask_scheduler(management_dir):
    print '-'*50
    print 'setting up dask-scheduler'
    sfile = _get_sfile(management_dir)
    command = 'screen -S dask_scheduler_test -dm bash -c "source ~/.bashrc; source_isf; dask-scheduler --port=9796 --scheduler-file={}"'
    command = 'dask-scheduler --scheduler-file={} &'
    command = command.format(sfile)
    print command
    os.system(command)
    print '-'*50

def setup_local_cluster():
    print '-'*50
    print 'setting up local dask-cluster'
    import psutil
    sfile = _get_sfile(management_dir) + '.' + str(process_number)
    command = 'dask-scheduler --scheduler-file={} --port=4321 &'
    command = command.format(sfile)
    print command
    os.system(command)
    n_cpus = psutil.cpu_count(logical=False)
    command = 'dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --memory-limit=100e9 &'.format(nprocs = n_cpus, sfile = sfile)
    print '-'*50
# In[7]:

#################################################
# setting up dask-worker
#################################################
def setup_dask_workers(management_dir):
    print '-'*50
    print 'setting up dask-workers'
    import psutil
    n_cpus = psutil.cpu_count(logical=False)
    sfile = _get_sfile(management_dir)
    command = 'screen -S dask_workers_test -dm bash -c "source ~/.bashrc; source_isf; ' +     'dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --memory-limit=10e25"'.format(nprocs = n_cpus, sfile = sfile)
    command = 'dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --memory-limit=100e9 &'.format(nprocs = n_cpus, sfile = sfile)

    print command
    os.system(command)
    print '-'*50

##############################################
# setting up jupyter-notebook
#############################################
def setup_jupyter_notebook():
    print '-'*50
    print 'setting up jupyter notebook'
    check_locking_config() 
    command = 'screen -S jupyter -dm bash -c "source ~/.bashrc; source_isf; ' +     '''jupyter-notebook --ip='*' --no-browser --port=11112"'''
    command = "cd notebooks; jupyter-notebook --ip='*' --no-browser --port=11112 &"
    print command
    os.system(command)    
    print '-'*50

# In[8]:

if process_number == 0:
    setup_locking_server()
    setup_dask_scheduler(management_dir)
    setup_jupyter_notebook()
setup_locking_config()
setup_dask_workers(management_dir)
setup_local_cluster()
#if not process_number == 0:
#    # compute job will be started from first process
#    time.sleep(60*60*24*365)
time.sleep(60*60*24*365)

#def iteratively_register_sub_mdbs(mdb):
#    mdb._register_this_database()
#    for k in mdb.keys():
#        if mdb.metadata[k]['dumper'] == 'just_create_mdb':
#            iteratively_register_sub_mdbs(mdb[k])
#            
#import Interface as I
#import simrun3.robust_dask_delayed_execution
#
#mdb = I.ModelDataBase('/axon/scratch/abast/results/20190726_network_embedding_with_random_synapse_locations_for_L5tt_models')
#iteratively_register_sub_mdbs(mdb)
#
#delayeds_db = mdb['delayeds_db']
#
#rde = simrun3.robust_dask_delayed_execution.RobustDaskDelayedExecution(delayeds_db)
#
#ds = rde.run_mdb()
#
#sfile =  _get_sfile(management_dir)
#client = I.distributed.Client(scheduler_file=sfile)
#
#futures = client.compute(ds)
#
#time.sleep(60*60*24*7)
### client.gather(futures)
