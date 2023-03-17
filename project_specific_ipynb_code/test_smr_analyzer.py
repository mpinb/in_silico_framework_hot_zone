import unittest


class Tests(unittest.TestCase):       
    def setUp(self):
        pass

    def test_generate_synapse_activation_returns_filelist(self):
        dirPrefix = tempfile.mkdtemp()
        try:
            dummy = simrun2.generate_synapse_activations.generate_synapse_activations(cellParamName, networkName, dirPrefix=dirPrefix, nSweeps=1, nprocs=1, tStop=345, silent=True)
            dummy = dummy.compute(get = synchronous_scheduler)
        except:
            raise
        finally:
            shutil.rmtree(dirPrefix)
        self.assertIsInstance(dummy[0][0][0], str)       