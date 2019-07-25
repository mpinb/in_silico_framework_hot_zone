
# coding: utf-8

# In[1]:

import fasteners
import os
import shutil
import sys


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
def setup_locking_server():
    print '-'*50
    print 'setting up locking server'
    command = 'screen -S redis_server_test -dm bash -c "source ~/.bashrc; source_isf; redis-server --save "" --appendonly no --port 8885 --protected-mode no"'
    print command
    os.system(command)
    print '-'*50

def setup_locking_config():
    print '-'*50
    print 'updating locking configuration to use new server'
    import model_data_base.distributed_lock

    import socket
    socket.gethostname()

    config = [dict(type = 'redis', config = dict(host = socket.gethostname(), port = 8885, socket_timeout = 1))]

    print 'old locking configuration'
    print model_data_base.distributed_lock.server
    print model_data_base.distributed_lock.client

    import time
    while True:
        try:
            model_data_base.distributed_lock.update_config(config)
            break
        except RuntimeError:
            print 'could not connect, retrying in 1 sec'
            time.sleep(1) 

    print 'updated locking configuration'
    print model_data_base.distributed_lock.server
    print model_data_base.distributed_lock.client
    print '-'*50


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
    command = command.format(sfile)
    print command
    os.system(command)
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
    command = 'screen -S dask_workers_test -dm bash -c "source ~/.bashrc; source_isf; ' +     'dask-worker --nthreads 1  --nprocs {nprocs} --scheduler-file={sfile} --memory-limit=100e9"'.format(nprocs = n_cpus, sfile = sfile)
    print command
    os.system(command)
    print '-'*50


# In[8]:

if process_number == 0:
    setup_locking_server()
    setup_locking_config()
    setup_dask_scheduler(management_dir)
setup_dask_workers(management_dir)


import time
<<<<<<< HEAD
time.sleep(60*60*24*365)
=======
time.sleep(300)
>>>>>>> 53e2a97dd874064a568f8b1fac7db105137838c3
