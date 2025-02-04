import mechanisms.l5pt as l5pt_mechanisms
import os
import numpy as np
from biophysics_fitting.hay_complete_default_setup import get_Simulator
from biophysics_fitting.L5tt_parameter_setup import get_L5tt_template_v2

PARENT_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_FOLDER = os.path.join(PARENT_DIR, "data")

def scale_apical(cell_param, params):
    assert len(params) == 1
    cell_param.cell_modify_functions.scale_apical.scale = params['scale']
    return cell_param

def get_test_simulator_morph89(step=True):
    fixed_params = get_test_fixed_params_morph89()
    s = get_Simulator(fixed_params, step=step)
    s.setup.cell_param_generator = get_L5tt_template_v2
    s.setup.cell_param_modify_funs.append(('scale_apical', scale_apical))
    return s

def get_test_fixed_params_morph89():
    p = {'BAC.hay_measure.recSite': 294.8203371921156,
    'BAC.stim.dist': 294.8203371921156,
    'bAP.hay_measure.recSite1': 294.8203371921156,
    'bAP.hay_measure.recSite2': 474.8203371921156,
    'hot_zone.min_': 384.8203371921156,
    'hot_zone.max_': 584.8203371921156,
    'hot_zone.outsidescale_sections': [23,
                                        24,
                                        25,
                                        26,
                                        27,
                                        28,
                                        29,
                                        31,
                                        32,
                                        33,
                                        34,
                                        35,
                                        37,
                                        38,
                                        40,
                                        42,
                                        43,
                                        44,
                                        46,
                                        48,
                                        50,
                                        51,
                                        52,
                                        54,
                                        56,
                                        58,
                                        60],
    'morphology.filename': os.path.join(
        TEST_DATA_FOLDER, '89_L5_CDK20050712_nr6L5B_dend_PC_neuron_transform_registered_C2.hoc')}
    return p

def serialize_voltage_traces(v):
    # make json serializable
    for stim in v:
        v[stim]["tVec"] = list(v[stim]["tVec"])
        serializable_vlist = []
        for vList in v[stim]["vList"]:
            # convert each sublist
            serializable_vlist.append(list(vList))
        v[stim]["vList"] = serializable_vlist
    return v

def deserialize_voltage_traces(v):
    for stim in v:
        v[stim]["tVec"] = np.array(v[stim]["tVec"])
        for i, vList in enumerate(v[stim]["vList"]):
            v[stim]["vList"][i] = np.array(vList)
        v[stim]["vList"] = np.array(v[stim]["vList"])
    return v