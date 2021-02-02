'''
Created on Nov 08, 2018

@author: abast
'''
if six.PY2:
    from __future__ import absolute_import

# from functools import partial
# import numpy as np
# import pandas as pd
# 
# import single_cell_parser as scp
# 
# from . import setup_stim
# from .utils import tVec, vmSoma, vmApical
# from .parameters import set_fixed_params, param_to_kwargs
from .parameters import param_to_kwargs
# from .simulator import Simulator, run_fun
# from .L5tt_parameter_setup import get_L5tt_template, set_morphology, set_ephys, set_hot_zone, set_param, set_many_param
# # moved to bottom to resolve circular import
# # from .hay_evaluation import hay_evaluate_BAC, hay_evaluate_bAP, hay_evaluate_StepOne, hay_evaluate_StepTwo, hay_evaluate_StepThree
# 
from .evaluator import Evaluator
from toolz.dicttoolz import merge
# 
# from .combiner import Combiner


from . import hay_evaluation_python
from .utils import tVec, vmSoma, vmApical, vmMax


################################################
# Simulator
################################################

def record_bAP(cell, recSite1 = None, recSite2 = None):
    assert(recSite1 is not None)
    assert(recSite2 is not None)
    return {'tVec': tVec(cell), 
            'vList': (vmSoma(cell), vmApical(cell, recSite1), vmApical(cell, recSite2)),
            'vMax': vmMax(cell)}

def record_BAC(cell, recSite = None):
    return {'tVec': tVec(cell), 
            'vList': (vmSoma(cell), vmApical(cell, recSite)),
            'vMax': vmMax(cell)}

def record_Step(cell):
    return {'tVec': tVec(cell), 
            'vList': [vmSoma(cell)],
            'vMax': vmMax(cell)}


def get_Simulator(fixed_params, step = False, vInit = False):
    s = hay_complete_default_setup.get_Simulator(fixed_params, step = step)
    s.setup.stim_response_measure_funs = []
    s.setup.stim_response_measure_funs.append(['bAP.hay_measure', param_to_kwargs(record_bAP)])
    s.setup.stim_response_measure_funs.append(['BAC.hay_measure', param_to_kwargs(record_BAC)])
    if vInit:
        raise NotImplementedError
    return s

######################################################
# Evaluator
######################################################

def get_Evaluator(step = False, vInit = False, bAP_kwargs = {}, BAC_kwargs = {}):
    e = Evaluator()
    bap = hay_evaluation_python.bAP(**bAP_kwargs)
    bac = hay_evaluation_python.BAC(**BAC_kwargs)

    e.setup.evaluate_funs.append(['BAC.hay_measure', 
                                  bac.get,
                                  'BAC.hay_features'])

    e.setup.evaluate_funs.append(['bAP.hay_measure',
                                  bap.get,
                                  'bAP.hay_features'])

    if step:
        raise NotImplementedError
    if vInit:
        raise NotImplementedError
    e.setup.finalize_funs.append(lambda x: merge(list(x.values())))    
    return e

##############################################################
# Combiner
##############################################################

def get_Combiner(step = False):
    return hay_complete_default_setup.getCombiner(step = step)

from . import hay_complete_default_setup