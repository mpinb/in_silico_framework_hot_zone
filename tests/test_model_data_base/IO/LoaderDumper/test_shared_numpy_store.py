from ...context import *
from model_data_base.model_data_base import ModelDataBase
from ... import decorators
import numpy as np
import unittest
import signal
import time
from multiprocessing import Process

from  model_data_base.IO.LoaderDumper.shared_numpy_store import *

class TemporaryDirectory: # just for testing
    def __init__(self, suffix="", prefix="tmp", dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir

    def __enter__(self):
        self.name = tempfile.mkdtemp(suffix=self.suffix, prefix=self.prefix, dir=self.dir)
        return self.name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.name)



def interruptible_task():
    import time
    print("Starting interruptible task...")
    for i in range(5):
        print("Interruptible task progress: {}/5".format(i+1))
        time.sleep(1/5)
    print("Interruptible task completed.")


def uninterruptible_task():
    import time
    with Uninterruptible():
        print("Starting uninterruptible task...")
        for i in range(5):
            print("Uninterruptible task progress: {}/5".format(i+1))
            time.sleep(1/5)
        print("Uninterruptible task completed.")
        
        


class TestSharedNumpyStore(unittest.TestCase):
    def test_shared_array_functions(self):
        arr = np.array([1, 2, 3])
        buffer, shared_array = shared_array_from_numpy(arr, name=None)
        shm, shared_arr_from_name = shared_array_from_shared_mem_name(buffer.name, dtype=arr.dtype, shape=arr.shape)
        assert np.array_equal(arr, shared_arr_from_name)    
        shm.close()    
        shm.unlink()        
        
        
    def test_SharedNumpyStore(self):
        arr = np.array([1, 2, 3])
        with TemporaryDirectory() as tempdir:
            nps = SharedNumpyStore(tempdir)
            nps.save(arr, 'testarray')
            buffer, shared_array = nps.load('testarray', load_from_disk = True)
            assert np.array_equal(arr, shared_array)
            nps.close()
            
    def test_append_save(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[7, 8, 9], [10, 11, 12]])
        combined_arr = np.concatenate((arr1, arr2), axis=0)

        with TemporaryDirectory() as tempdir:
            nps = SharedNumpyStore(tempdir)

            nps.save(arr1, 'testarray')
            nps.append_save(arr2, 'testarray')

            _, shared_array = nps.load('testarray', load_from_disk=True)

            print("Expected combined array:")
            print(combined_arr)

            print("Loaded array from store:")
            print(shared_array)

            assert np.array_equal(combined_arr, shared_array)  
            nps.close()
            
            
    def test_append_save_no_flush_leaves_array_unchanged(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[7, 8, 9], [10, 11, 12]])
        combined_arr = np.concatenate((arr1, arr2), axis=0)

        with TemporaryDirectory() as tempdir:
            nps = SharedNumpyStore(tempdir)

            nps.save(arr1, 'testarray')
            nps.append_save(arr2, 'testarray', autoflush = False)

            _, shared_array = nps.load('testarray', load_from_disk=True)

            print("Expected combined array:")
            print(combined_arr)

            print("Loaded array from store:")
            print(shared_array)

            assert np.array_equal(arr1, shared_array) 
            nps.close()
            
    def test_robustness(self):
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[7, 8, 9], [10, 11, 12]])
        combined_arr = np.vstack((arr1, arr2))

        with TemporaryDirectory() as tempdir:
            nps = SharedNumpyStore(tempdir)

            # Save and append arrays
            nps.save(arr1, 'testarray')
            
            # Append 10 random bytes at the end of the file
            fname, shape, dtype = nps._get_metadata_from_fname(nps._files['testarray'])
            file_path = os.path.join(tempdir, fname)
            print(file_path)            
            with open(file_path, 'ab') as f:
                f.write(os.urandom(10))
            
            # write another array
            nps.append_save(arr2, 'testarray')

            # Append 10 random bytes at the end of the file
            fname, shape, dtype = nps._get_metadata_from_fname(nps._files['testarray'])
            file_path = os.path.join(tempdir, fname)
            print(file_path)
            with open(file_path, 'ab') as f:
                f.write(os.urandom(10))

            # Load the array and check if it still works correctly
            _, shared_array = nps.load('testarray', load_from_disk=True)
            assert np.array_equal(combined_arr, shared_array)

            nps.close()        
            
            
    def test_uninterruptible(self):
        print("Running interruptible task in a separate process.")
        t0 = time.time()        
        interruptible_process = Process(target=interruptible_task)
        interruptible_process.start()
        print("Sending SIGTERM to interruptible task.")
        time.sleep(2/5)
        os.kill(interruptible_process.pid, signal.SIGTERM)
        interruptible_process.join()
        t_i = time.time() - t0
        print('interuptible process was running for', t_i, 'seconds')

        print("Running uninterruptible task in a separate process.")
        t0 = time.time()        
        uninterruptible_process = Process(target=uninterruptible_task)
        uninterruptible_process.start()
        time.sleep(2/5)
        print("Sending SIGTERM to uninterruptible task.")
        os.kill(uninterruptible_process.pid, signal.SIGTERM)
        uninterruptible_process.join()
        t_ni = time.time() - t0
        print('uninteruptible process was running for', time.time() - t0, 'seconds')
        
        assert(t_i > 2/5)
        assert(t_i < 5/5)
        assert(t_ni > 5/5)
        
        
        
        
#test_SharedNumpyStore()
#test_shared_array_functions()    