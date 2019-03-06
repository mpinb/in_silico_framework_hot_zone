import Interface as I
import decorators
import unittest
import context

import simrun3.synaptic_strength_fitting 
reload(simrun3.synaptic_strength_fitting ) 
PSPs = simrun3.synaptic_strength_fitting.PSPs

# class FakeFuture():
#     def __init__(self, f):
#         self.f = f
#     def result(self):
#         return self.f
#     
# class FakeClient():
#     def compute(self, *args, **kwargs):
#         res = I.dask.compute(*args, get = I.dask.get)
#         return [FakeFuture(r) for r in res]
    
class Tests(unittest.TestCase):       
    def setUp(self):
        pass

    @decorators.testlevel(2)    
    def test_VPM_synaptic_strength_is_between_1_75_and_1_85(self):
        PSPs = simrun3.synaptic_strength_fitting.PSPs
        confile = I.os.path.join(context.data_dir, '86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2_synapses_20150202-1834_4335.con')
        neuron_param = I.os.path.join(context.data_dir, 'neuron_model.param')
        neuron_param = I.scp.build_parameters(neuron_param)
        neuron_param.neuron['cell_modify_functions'] = I.scp.NTParameterSet({'scale_apical_morph_86': {}})
        psps = PSPs(confile=confile, neuron_param=neuron_param)
        indexes = [lv for lv, k in enumerate(psps._keys) if k[0] == 'VPM_C2']
        psps._delayeds = [psps._delayeds[index] for index in indexes]
        psps._keys = [psps._keys[index] for index in indexes]
        c = I.distributed.Client('localhost:8786') # c = FakeClient()
        psps.run(c)
        optimal_g_pdf = psps.get_optimal_g(I.barrel_cortex.get_EPSP_measurement())
        gVPM = optimal_g_pdf.loc['VPM_C2']['optimal g']
        self.assertGreaterEqual(gVPM, 1.75)
        self.assertGreaterEqual(1.85, gVPM)        