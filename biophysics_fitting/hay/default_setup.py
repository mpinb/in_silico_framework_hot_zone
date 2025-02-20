"""
A Python translation of the setup for in-silico current injection experiments as described in :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`.
"""

from __future__ import absolute_import

import os
from functools import partial

import numpy as np
import pandas as pd
from toolz.dicttoolz import merge

import single_cell_parser as scp

from .. import setup_stim
from ..combiner import Combiner
from ..evaluator import Evaluator
from ..L5tt_parameter_setup import (
    get_L5tt_template,
    get_L5tt_template_v2,
    set_ephys,
    set_hot_zone,
    set_many_param,
    set_morphology,
    set_param,
)
from ..parameters import param_to_kwargs, set_fixed_params
from ..simulator import Simulator, run_fun
from ..utils import tVec, vmApical, vmMax, vmSoma
from . import evaluation

__author__ = "Arco Bast"
__date__ = "2018-11-08"

################################################
# Simulator
################################################


def record_bAP(cell, recSite1=None, recSite2=None):
    """Extract the voltage traces from the soma and two apical dendritic locations.

    This is used to quantify the voltage trace of a backpropagating AP (bAP)
    stimulus in a pyramidal neuron. The two apical recording sites are used
    to calculate e.g. backpropagating attenuation.

    Args:
        cell (:py:class:`~single_cell_parser.cell.Cell`): The cell object.
        recSite1 (float): The distance (um) from the soma to the first recording site.
        recSite2 (float): The distance (um) from the soma to the second recording site.

    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    assert recSite1 is not None
    assert recSite2 is not None
    return {
        "tVec": tVec(cell),
        "vList": (vmSoma(cell), vmApical(cell, recSite1), vmApical(cell, recSite2)),
        "vMax": vmMax(cell),
    }


def record_BAC(cell, recSite=None):
    """Extract the voltage traces from the soma and an apical dendritic location.

    This is used to quantify the voltage trace of a bAP-Activated Ca2+ (BAC) stimulus

    Args:
        cell (:py:class:`~single_cell_parser.cell.Cell`): The cell object.
        recSite (float): The distance (um) from the soma to the apical recording site.

    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    return {
        "tVec": tVec(cell),
        "vList": (vmSoma(cell), vmApical(cell, recSite)),
        "vMax": vmMax(cell),
    }


def record_Step(cell):
    """Extract the voltage trace from the soma.

    This is used to quantify the response of the cell to step currents.

    Args:
        cell (:py:class:`~single_cell_parser.cell.Cell`): The cell object.

    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    return {"tVec": tVec(cell), "vList": [vmSoma(cell)], "vMax": vmMax(cell)}


def get_Simulator(fixed_params, step=False, vInit=False):
    """Get a set up :py:class:`~biophysics_fitting.simulator.Simulator` object for the Hay protocol.

    Given cell-specific fixed parameters, set up a simulator object for the Hay protocol,
    including measuring functions for bAP and BAC stimuli (no step currents)

    Args:
        fixed_params (dict): A dictionary of fixed parameters for the cell.
        step (bool): Whether to include step current measurements. These take quite long to simulate. Default: ``False``.
        vInit (bool): Whether to include vInit measurements. (not implemented yet)

    Returns:
        :py:class:`~biophysics_fitting.simulator.Simulator`: A simulator object.

    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    s = Simulator()
    s.setup.stim_response_measure_funs = []
    s.setup.cell_param_generator = get_L5tt_template
    s.setup.params_modify_funs.append(
        ["fixed_params", partial(set_fixed_params, fixed_params=fixed_params)]
    )
    s.setup.cell_param_modify_funs.append(
        ["morphology", param_to_kwargs(set_morphology)]
    )
    s.setup.cell_param_modify_funs.append(["ephys", set_ephys])
    s.setup.cell_param_modify_funs.append(["params", set_param])
    s.setup.cell_param_modify_funs.append(["many_params", set_many_param])

    s.setup.cell_param_modify_funs.append(["hot_zone", param_to_kwargs(set_hot_zone)])
    s.setup.cell_generator = scp.create_cell
    # s.setup.cell_modify_funs.append('apical_dendrite_scaling', apical_dendrite_scaling)

    # --- Stimulus setup functions
    s.setup.stim_setup_funs.append(["bAP.stim", param_to_kwargs(setup_stim.setup_bAP)])
    s.setup.stim_setup_funs.append(["BAC.stim", param_to_kwargs(setup_stim.setup_BAC)])
    if step:
        s.setup.stim_setup_funs.append(
            ["StepOne.stim", param_to_kwargs(setup_stim.setup_StepOne)]
        )
        s.setup.stim_setup_funs.append(
            ["StepTwo.stim", param_to_kwargs(setup_stim.setup_StepTwo)]
        )
        s.setup.stim_setup_funs.append(
            ["StepThree.stim", param_to_kwargs(setup_stim.setup_StepThree)]
        )

    # --- Stimulus run functions
    ## bAP and BAC
    run_fun_bAP_BAC = partial(
        run_fun,
        T=34.0,
        Vinit=-75.0,
        dt=0.025,
        recordingSites=[],
        tStart=0.0,
        tStop=600.0,
        vardt=True,
    )
    s.setup.stim_run_funs.append(["bAP.run", param_to_kwargs(run_fun_bAP_BAC)])
    s.setup.stim_run_funs.append(["BAC.run", param_to_kwargs(run_fun_bAP_BAC)])

    ## Step currents
    run_fun_Step = partial(
        run_fun,
        T=34.0,
        Vinit=-75.0,
        dt=0.025,
        recordingSites=[],
        tStart=0.0,
        tStop=3000.0,
        vardt=True,
    )

    if step:
        s.setup.stim_run_funs.append(["StepOne.run", param_to_kwargs(run_fun_Step)])
        s.setup.stim_run_funs.append(["StepTwo.run", param_to_kwargs(run_fun_Step)])
        s.setup.stim_run_funs.append(["StepThree.run", param_to_kwargs(run_fun_Step)])
    s.setup.stim_response_measure_funs.append(
        ["bAP.hay_measure", param_to_kwargs(record_bAP)]
    )
    s.setup.stim_response_measure_funs.append(
        ["BAC.hay_measure", param_to_kwargs(record_BAC)]
    )
    if step:
        s.setup.stim_response_measure_funs.append(
            ["StepOne.hay_measure", param_to_kwargs(record_Step)]
        )
        s.setup.stim_response_measure_funs.append(
            ["StepTwo.hay_measure", param_to_kwargs(record_Step)]
        )
        s.setup.stim_response_measure_funs.append(
            ["StepThree.hay_measure", param_to_kwargs(record_Step)]
        )
    if vInit:
        raise NotImplementedError
    return s


######################################################
# Evaluator
######################################################
def interpolate_vt(voltage_trace_):
    """Interpolate a voltage trace so that is has fixed time interval

    The NEURON simulator allows for a variable time step, which can make
    comparing voltage traces difficult. This function interpolates the voltage
    traces so that they have a fixed time interval of 0.025 ms.

    Args:
        voltage_trace_ (dict): A dictionary of voltage traces.

    Returns:
        dict: A dictionary of voltage traces with a fixed time interval.
    """
    out = {}
    for k in voltage_trace_:
        t = voltage_trace_[k]["tVec"]
        t_new = np.arange(0, max(t), 0.025)
        vList_new = [
            np.interp(t_new, t, v) for v in voltage_trace_[k]["vList"]
        ]  # I.np.interp
        out[k] = {"tVec": t_new, "vList": vList_new}
        if "iList" in voltage_trace_[k]:
            iList_new = [np.interp(t_new, t, i) for i in voltage_trace_[k]["iList"]]
            out[k] = {"tVec": t_new, "vList": vList_new, "iList": iList_new}
    return out


def map_truefalse_to_str(dict_):
    """Convert True/False to 'True'/'False' in a dictionary

    Args:
        dict_ (dict): A dictionary with boolean values.

    Returns:
        dict: A dictionary with boolean values converted to strings.
    """

    def _helper(x):
        if (x is True) or (x is np.True_):
            return "True"
        elif (x is False) or (x is np.False_):
            return "False"
        else:
            return x

    return {k: _helper(dict_[k]) for k in dict_}


def get_Evaluator(
    step=False,
    vInit=False,
    bAP_kwargs={},
    BAC_kwargs={},
    StepOne_kwargs={},
    StepTwo_kwargs={},
    StepThree_kwargs={},
    interpolate_voltage_trace=True,
):
    """Get a :py:class:`~biophysics_fitting.evaluator.Evaluator` object for the Hay protocol.

    Sets up an evaluator object for the Hay protocol, including measuring functions for bAP, BAC and three step current stimuli.

    Args:
        step (bool): Whether to include step current measurements (not implemented yet).
        vInit (bool): Whether to include vInit measurements. (not implemented yet)
        bAP_kwargs (dict): Keyword arguments for the ``bAP`` measurement function.
        BAC_kwargs (dict): Keyword arguments for the ``BAC`` measurement function.
        StepOne_kwargs (dict): Keyword arguments for the ``StepOne`` measurement function.
        StepTwo_kwargs (dict): Keyword arguments for the ``StepTwo`` measurement function.
        StepThree_kwargs (dict): Keyword arguments for the ``StepThree`` measurement function.
        interpolate_voltage_trace (bool): Whether to interpolate the voltage trace to a fixed time interval.

    Returns:
        :py:class:`~biophysics_fitting.evaluator.Evaluator`: An evaluator object.

    Raises:
        NotImplementedError: If :paramref:vInit is set to True.

    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    e = Evaluator()
    bap = evaluation.bAP(**bAP_kwargs)
    bac = evaluation.BAC(**BAC_kwargs)

    if interpolate_voltage_trace:
        e.setup.pre_funs.append(interpolate_vt)

    e.setup.evaluate_funs.append(["BAC.hay_measure", bac.get, "BAC.hay_features"])

    e.setup.evaluate_funs.append(["bAP.hay_measure", bap.get, "bAP.hay_features"])

    if step:
        step_one = evaluation.StepOne(**StepOne_kwargs)
        step_two = evaluation.StepTwo(**StepTwo_kwargs)
        step_three = evaluation.StepThree(**StepThree_kwargs)
        e.setup.evaluate_funs.append(
            ["StepOne.hay_measure", step_one.get, "StepOne.hay_features"]
        )
        e.setup.evaluate_funs.append(
            ["StepTwo.hay_measure", step_two.get, "StepTwo.hay_features"]
        )
        e.setup.evaluate_funs.append(
            ["StepThree.hay_measure", step_three.get, "StepThree.hay_features"]
        )
    if vInit:
        raise NotImplementedError
    e.setup.finalize_funs.append(lambda x: merge(list(x.values())))
    e.setup.finalize_funs.append(map_truefalse_to_str)

    return e


##############################################################
# Combiner
##############################################################


def get_Combiner(step=False, include_DI3=True):
    """Get a set up :py:class:`~biophysics_fitting.combiner.Combiner` object for the Hay protocol.

    Args:
        step (bool): Whether to include step current measurements.
        include_DI3 (bool):
            Whether to include the doublet ISI for the Step3 protocol in the step current measurements.
            Default: ``True``.

    Returns:
        :py:class:`~biophysics_fitting.combiner.Combiner`: A combiner object.

    See also:
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    # up to 20220325, DI3 has not been included and was not in the fit_features file.
    c = Combiner()
    c.setup.append(
        "bAP_somatic_spike", ["bAP_APwidth", "bAP_APheight", "bAP_spikecount"]
    )
    c.setup.append("bAP", ["bAP_att2", "bAP_att3"])
    c.setup.append("BAC_somatic", ["BAC_ahpdepth", "BAC_APheight", "BAC_ISI"])
    c.setup.append("BAC_caSpike", ["BAC_caSpike_height", "BAC_caSpike_width"])
    c.setup.append("BAC_spikecount", ["BAC_spikecount"])
    if step:
        c.setup.append("step_mean_frequency", ["mf1", "mf2", "mf3"])
        c.setup.append(
            "step_AI_ISIcv", ["AI1", "AI2", "ISIcv1", "ISIcv2", "AI3", "ISIcv3"]
        )
        if include_DI3:
            c.setup.append("step_doublet_ISI", ["DI1", "DI2", "DI3"])
        else:
            c.setup.append("step_doublet_ISI", ["DI1", "DI2"])
        c.setup.append("step_AP_height", ["APh1", "APh2", "APh3"])
        c.setup.append("step_time_to_first_spike", ["TTFS1", "TTFS2", "TTFS3"])
        c.setup.append(
            "step_AHP_depth",
            ["fAHPd1", "fAHPd2", "fAHPd3", "sAHPd1", "sAHPd2", "sAHPd3"],
        )
        c.setup.append("step_AHP_slow_time", ["sAHPt1", "sAHPt2", "sAHPt3"])
        c.setup.append("step_AP_width", ["APw1", "APw2", "APw3"])
    c.setup.combinefun = np.max
    return c


def get_fixed_params_example():
    """Get an example of cell-specific fixed params.

    Fixed parameters are parameters that are required for a stimulus protocol.
    They are specific to a stimulus protocol, and to the morphology of the cell.
    This method provides an example of such fixed parameters for a L5PT and the
    bAP and BAC stimuli.

    Returns:
        dict: A dictionary with the fixed parameters.
    """
    fixed_params = {
        "hot_zone.max_": 614,
        "hot_zone.min_": 414,
        "bAP.hay_measure.recSite1": 349,
        "bAP.hay_measure.recSite2": 529,
        "BAC.stim.dist": 349,
        "BAC.hay_measure.recSite": 349,
        "morphology.filename": os.path.abspath(
            os.path.join(
                __file__,
                "..",
                "..",
                "getting_started",
                "example_data",
                "morphology",
                "89_L5_CDK20050712_nr6L5B_dend_PC_neuron_transform_registered_C2.hoc",
            )
        ),
    }  # TODO: fix fixed params morphology example
    return fixed_params


def get_hay_objective_names():
    """The objective names for the Hay stimulus protocol.

    Returns:
        list: A list of the objective names.

    Note:
        The objectives are specific to the L5PT and the Hay stimulus protocol.
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    return [
        "bAP_APwidth",
        "bAP_APheight",
        "bAP_spikecount",
        "bAP_att2",
        "bAP_att3",
        "BAC_ahpdepth",
        "BAC_APheight",
        "BAC_ISI",
        "BAC_caSpike_height",
        "BAC_caSpike_width",
        "BAC_spikecount",
        "mf1",
        "mf2",
        "mf3",
        "AI1",
        "AI2",
        "ISIcv1",
        "ISIcv2",
        "AI3",
        "ISIcv3",
        "DI1",
        "DI2",
        "DI3",
        "APh1",
        "APh2",
        "APh3",
        "TTFS1",
        "TTFS2",
        "TTFS3",
        "fAHPd1",
        "fAHPd2",
        "fAHPd3",
        "sAHPd1",
        "sAHPd2",
        "sAHPd3",
        "sAHPt1",
        "sAHPt2",
        "sAHPt3",
        "APw1",
        "APw2",
        "APw3",
    ]


def get_hay_param_names():
    """The parameter names for the Hay stimulus protocol.

    Contains all the biophysical parameters for a L5PT as considered in the Hay stimulus protocol.

    Returns:
        list: A list of the parameter names.

    Note:
        The parameters are specific to the L5PT and the Hay stimulus protocol.
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011` for more information.
    """
    return [
        "NaTa_t.soma.gNaTa_tbar",
        "Nap_Et2.soma.gNap_Et2bar",
        "K_Pst.soma.gK_Pstbar",
        "K_Tst.soma.gK_Tstbar",
        "SK_E2.soma.gSK_E2bar",
        "SKv3_1.soma.gSKv3_1bar",
        "Ca_HVA.soma.gCa_HVAbar",
        "Ca_LVAst.soma.gCa_LVAstbar",
        "CaDynamics_E2.soma.gamma",
        "CaDynamics_E2.soma.decay",
        "NaTa_t.axon.gNaTa_tbar",
        "Nap_Et2.axon.gNap_Et2bar",
        "K_Pst.axon.gK_Pstbar",
        "K_Tst.axon.gK_Tstbar",
        "SK_E2.axon.gSK_E2bar",
        "SKv3_1.axon.gSKv3_1bar",
        "Ca_HVA.axon.gCa_HVAbar",
        "Ca_LVAst.axon.gCa_LVAstbar",
        "CaDynamics_E2.axon.gamma",
        "CaDynamics_E2.axon.decay",
        "none.soma.g_pas",
        "none.axon.g_pas",
        "none.dend.g_pas",
        "none.apic.g_pas",
        "Im.apic.gImbar",
        "NaTa_t.apic.gNaTa_tbar",
        "SKv3_1.apic.gSKv3_1bar",
        "Ca_HVA.apic.gCa_HVAbar",
        "Ca_LVAst.apic.gCa_LVAstbar",
        "SK_E2.apic.gSK_E2bar",
        "CaDynamics_E2.apic.gamma",
        "CaDynamics_E2.apic.decay",
    ]


def get_hay_params_pdf():
    """Get the ranges for the biophysical parameters of an L5PT.

    conductance parameters (gbar or gpas) are in units of [S/cm^2]
    tau parameters are in units of [ms]
    gamma parameters are a fraction and unitless

    Returns:
        pd.DataFrame: A DataFrame with the parameter names as index and the ranges as columns.

    Note:
        The parameters are specific to the L5PT and the Hay stimulus protocol.
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`."""
    d = {
        "max": [
            4.0,
            0.01,
            1.0,
            0.1,
            0.1,
            2.0,
            0.001,
            0.01,
            0.05,
            1000.0,
            4.0,
            0.01,
            1.0,
            0.1,
            0.1,
            2.0,
            0.001,
            0.01,
            0.05,
            1000.0,
            5.02e-05,
            5.0e-05,
            0.0001,
            0.0001,
            0.001,
            0.04,
            0.04,
            0.005,
            0.2,
            0.01,
            0.05,
            200.0,
        ],
        "min": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0005,
            20.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0005,
            20.0,
            2.0e-05,
            2.0e-05,
            3.0e-05,
            3.0e-05,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0005,
            20.0,
        ],
        "param_names": get_hay_param_names(),
    }
    return pd.DataFrame(d).set_index("param_names", drop=True)[["min", "max"]]


def get_hay_problem_description():
    """Get the problem description for the Hay stimulus protocol.

    This method returns a dataframe that contains the objective names,
    the average value per objective, and the std per objective.
    Each objective is associated with a specific input stimulus, given in the 'stim_name' column.
    This distribution of parametrized responses can be used to quantify a cell's response to a specific stimulus.

    Returns:
        pd.DataFrame: A DataFrame with the problem description.

    Note:
        The objectives are specific to the L5PT and the Hay stimulus protocol.
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`.
    """
    d = {
        "feature": {
            0: "Spikecount",
            1: "AP_height",
            2: "AP_width",
            3: "BPAPatt2",
            4: "BPAPatt3",
            5: "AHP_depth_abs",
            6: "AP_height",
            7: "BAC_ISI",
            8: "BAC_caSpike_height",
            9: "BAC_caSpike_width",
            10: "Spikecount",
            11: "mean_frequency2",
            12: "adaptation_index2",
            13: "ISI_CV",
            14: "doublet_ISI",
            15: "time_to_first_spike",
            16: "AP_height",
            17: "AHP_depth_abs_fast",
            18: "AHP_depth_abs_slow",
            19: "AHP_slow_time",
            20: "AP_width",
            21: "mean_frequency2",
            22: "adaptation_index2",
            23: "ISI_CV",
            24: "doublet_ISI",
            25: "time_to_first_spike",
            26: "AP_height",
            27: "AHP_depth_abs_fast",
            28: "AHP_depth_abs_slow",
            29: "AHP_slow_time",
            30: "AP_width",
            31: "mean_frequency2",
            32: "adaptation_index2",
            33: "ISI_CV",
            34: "doublet_ISI",
            35: "time_to_first_spike",
            36: "AP_height",
            37: "AHP_depth_abs_fast",
            38: "AHP_depth_abs_slow",
            39: "AHP_slow_time",
            40: "AP_width",
        },
        "mean": {
            0: 1.0,
            1: 25.0,
            2: 2.0,
            3: 45.0,
            4: 36.0,
            5: -65.0,
            6: 25.0,
            7: 9.9009999999999998,
            8: 6.7300000000000004,
            9: 37.43,
            10: 3.0,
            11: 9.0,
            12: 0.0035999999999999999,
            13: 0.12039999999999999,
            14: 57.75,
            15: 43.25,
            16: 26.227399999999999,
            17: -51.951099999999997,
            18: -58.0443,
            19: 0.23760000000000001,
            20: 1.3077000000000001,
            21: 14.5,
            22: 0.0023,
            23: 0.10829999999999999,
            24: 6.625,
            25: 19.125,
            26: 16.520900000000001,
            27: -54.194899999999997,
            28: -60.512900000000002,
            29: 0.2787,
            30: 1.3833,
            31: 22.5,
            32: 0.0045999999999999999,
            33: 0.095399999999999999,
            34: 5.38,
            35: 7.25,
            36: 16.436800000000002,
            37: -56.557899999999997,
            38: -59.9923,
            39: 0.21310000000000001,
            40: 1.8647,
        },
        "objective": {
            0: "bAP_spikecount",
            1: "bAP_APheight",
            2: "bAP_APwidth",
            3: "bAP_att2",
            4: "bAP_att3",
            5: "BAC_ahpdepth",
            6: "BAC_APheight",
            7: "BAC_ISI",
            8: "BAC_caSpike_height",
            9: "BAC_caSpike_width",
            10: "BAC_spikecount",
            11: "mf1",
            12: "AI1",
            13: "ISIcv1",
            14: "DI1",
            15: "TTFS1",
            16: "APh1",
            17: "fAHPd1",
            18: "sAHPd1",
            19: "sAHPt1",
            20: "APw1",
            21: "mf2",
            22: "AI2",
            23: "ISIcv2",
            24: "DI2",
            25: "TTFS2",
            26: "APh2",
            27: "fAHPd2",
            28: "sAHPd2",
            29: "sAHPt2",
            30: "APw2",
            31: "mf3",
            32: "AI3",
            33: "ISIcv3",
            34: "DI3",
            35: "TTFS3",
            36: "APh3",
            37: "fAHPd3",
            38: "sAHPd3",
            39: "sAHPt3",
            40: "APw3",
        },
        "std": {
            0: 0.01,
            1: 5.0,
            2: 0.5,
            3: 10.0,
            4: 9.3300000000000001,
            5: 4.0,
            6: 5.0,
            7: 0.85170000000000001,
            8: 2.54,
            9: 1.27,
            10: 0.01,
            11: 0.88,
            12: 0.0091000000000000004,
            13: 0.032099999999999997,
            14: 33.479999999999997,
            15: 7.3200000000000003,
            16: 4.9702999999999999,
            17: 5.8212999999999999,
            18: 4.5814000000000004,
            19: 0.029899999999999999,
            20: 0.16650000000000001,
            21: 0.56000000000000005,
            22: 0.0055999999999999999,
            23: 0.036799999999999999,
            24: 8.6500000000000004,
            25: 7.3099999999999996,
            26: 6.1127000000000002,
            27: 5.5705999999999998,
            28: 4.6717000000000004,
            29: 0.026599999999999999,
            30: 0.2843,
            31: 2.2222,
            32: 0.0025999999999999999,
            33: 0.014,
            34: 0.83,
            35: 1.0,
            36: 6.9321999999999999,
            37: 3.5834000000000001,
            38: 3.9247000000000001,
            39: 0.036799999999999999,
            40: 0.41189999999999999,
        },
        "stim_name": {
            0: "bAP",
            1: "bAP",
            2: "bAP",
            3: "bAP",
            4: "bAP",
            5: "BAC",
            6: "BAC",
            7: "BAC",
            8: "BAC",
            9: "BAC",
            10: "BAC",
            11: "StepOne",
            12: "StepOne",
            13: "StepOne",
            14: "StepOne",
            15: "StepOne",
            16: "StepOne",
            17: "StepOne",
            18: "StepOne",
            19: "StepOne",
            20: "StepOne",
            21: "StepTwo",
            22: "StepTwo",
            23: "StepTwo",
            24: "StepTwo",
            25: "StepTwo",
            26: "StepTwo",
            27: "StepTwo",
            28: "StepTwo",
            29: "StepTwo",
            30: "StepTwo",
            31: "StepThree",
            32: "StepThree",
            33: "StepThree",
            34: "StepThree",
            35: "StepThree",
            36: "StepThree",
            37: "StepThree",
            38: "StepThree",
            39: "StepThree",
            40: "StepThree",
        },
        "stim_type": {
            0: "bAP",
            1: "bAP",
            2: "bAP",
            3: "bAP",
            4: "bAP",
            5: "BAC",
            6: "BAC",
            7: "BAC",
            8: "BAC",
            9: "BAC",
            10: "BAC",
            11: "SquarePulse",
            12: "SquarePulse",
            13: "SquarePulse",
            14: "SquarePulse",
            15: "SquarePulse",
            16: "SquarePulse",
            17: "SquarePulse",
            18: "SquarePulse",
            19: "SquarePulse",
            20: "SquarePulse",
            21: "SquarePulse",
            22: "SquarePulse",
            23: "SquarePulse",
            24: "SquarePulse",
            25: "SquarePulse",
            26: "SquarePulse",
            27: "SquarePulse",
            28: "SquarePulse",
            29: "SquarePulse",
            30: "SquarePulse",
            31: "SquarePulse",
            32: "SquarePulse",
            33: "SquarePulse",
            34: "SquarePulse",
            35: "SquarePulse",
            36: "SquarePulse",
            37: "SquarePulse",
            38: "SquarePulse",
            39: "SquarePulse",
            40: "SquarePulse",
        },
    }

    return pd.DataFrame.from_dict(d)


# def get_hay_problem_description():
#     setup_hay_evaluator()
#     sol = h.mc.get_sol()
#     stim_count = int(h.mc.get_stimulus_num())
#     stims = [sol.o(cur_stim).get_type().s for cur_stim in range(stim_count)]

#     objective_names = []
#     feature_types = []
#     stim_types = []
#     stim_names = []
#     mean = []
#     std = []
#     for lv in range(int(sol.count())):
#         for i in range (len(h.evaluator.stimulus_feature_type_list.o(lv))):
#             objective_names.append(h.evaluator.stimulus_feature_name_list.o(lv).o(i).s)
#             feature_types.append(h.evaluator.stimulus_feature_type_list.o(lv).o(i).s)
#             stim_types.append(sol.o(lv).get_type().s)
#             stim_names.append(sol.o(lv).get_name().s)
#             mean.append(h.evaluator.feature_mean_list.o(lv)[i])
#             std.append(h.evaluator.feature_std_list.o(lv)[i])
#     return I.pd.DataFrame(dict(objective = objective_names,
#                                feature = feature_types,
#                                stim_type = stim_types,
#                                stim_name = stim_names,
#                                mean = mean, std = std))[['objective', 'feature',
#                                                          'stim_name', 'stim_type',
#                                                          'mean', 'std']]

##############################################
# used to test reproducibility
##############################################


def get_feasible_model_params():
    """Get a set of feasible model parameters.

    This datframe contains biophysical parameters that are not only within range,
    but also likely values to find.
    They are however not guaranteed to yield realistic responses to a stimulus, when
    applied to a L5PT.
    In general, the response of a cell with these biophysical parameters
    heavily depends on the morphology as well.

    Conductance parameters (gbar or gpas) are in units of [S/cm^2]
    Tau parameters are time constants in units of [ms]
    Gamma parameters are a fraction and unitless

    Returns:
        pd.DataFrame: A DataFrame with the parameter names as index, feasible values as column 'x', and the min and max.

    Note:
        The parameters are specific to the L5PT and the Hay stimulus protocol.
        See :cite:t:`Hay_Hill_Schuermann_Markram_Segev_2011`.
    """
    pdf = get_hay_params_pdf()
    x = [
        1.971849,
        0.000363,
        0.008663,
        0.099860,
        0.073318,
        0.359781,
        0.000530,
        0.004958,
        0.000545,
        342.880108,
        3.755353,
        0.002518,
        0.025765,
        0.060558,
        0.082471,
        0.922328,
        0.000096,
        0.000032,
        0.005209,
        248.822554,
        0.000025,
        0.000047,
        0.000074,
        0.000039,
        0.000436,
        0.016033,
        0.008445,
        0.004921,
        0.003024,
        0.003099,
        0.0005,
        116.339356,
    ]
    pdf["x"] = x
    return pdf


def get_feasible_model_objectives():
    """Get a set of feasible values for the objectives of a biophysical model.

    Returns:
        pd.DataFrame: An example DataFrame with the objective names as index, feasible values as column 'y'.
    """
    pdf = get_hay_problem_description()
    index = get_hay_objective_names()
    y = [
        1.647,
        3.037,
        0.0,
        2.008,
        2.228,
        0.385,
        1.745,
        1.507,
        0.358,
        1.454,
        0.0,
        0.568,
        0.893,
        0.225,
        0.75,
        2.78,
        0.194,
        1.427,
        3.781,
        5.829,
        1.29,
        0.268,
        0.332,
        1.281,
        0.831,
        1.931,
        0.243,
        1.617,
        1.765,
        1.398,
        1.126,
        0.65,
        0.065,
        0.142,
        5.628,
        6.852,
        2.947,
        1.771,
        1.275,
        2.079,
    ]
    s = pd.Series(y, index=index)
    pdf.set_index("objective", drop=True, inplace=True)
    pdf["y"] = s
    return pdf
