import Interface as I
from . import decorators
import unittest
from . import context
import getting_started
import simrun3.synaptic_strength_fitting 
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
    
class TestSynapticStrengthFitting:

    @decorators.testlevel(2)    
    def test_VPM_synaptic_strength_is_between_1_75_and_1_85(self):
        PSPs = simrun3.synaptic_strength_fitting.PSPs
        confile = I.os.path.join(context.data_dir, '86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2_synapses_20150202-1834_4335.con')
        neuron_param = I.os.path.join(context.data_dir, 'neuron_model.param')
        neuron_param = I.scp.build_parameters(neuron_param)
        neuron_param.neuron['cell_modify_functions'] = I.scp.NTParameterSet({'scale_apical_morph_86': {}})
        filename = I.os.path.join(context.data_dir, "86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2.hoc")
        neuron_param.neuron['filename'] = filename
        psps = PSPs(confile=confile, neuron_param=neuron_param)
        indexes = [lv for lv, k in enumerate(psps._keys) if k[0] == 'VPM_C2']
        psps._delayeds = [psps._delayeds[index] for index in indexes]
        psps._keys = [psps._keys[index] for index in indexes]
        c = I.distributed.client_object_duck_typed # c = FakeClient()
        psps.run(c)
        optimal_g_pdf = psps.get_optimal_g(I.barrel_cortex.get_EPSP_measurement())
        gVPM = optimal_g_pdf.loc['VPM_C2']['optimal g']
        assert(1.85 >= gVPM >= 1.75)