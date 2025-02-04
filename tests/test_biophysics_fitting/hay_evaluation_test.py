import neuron
import numpy as np
import pytest

from biophysics_fitting.hay_complete_default_setup import (
    get_Evaluator as get_hay_Evaluator,
)
from biophysics_fitting.hay_complete_default_setup_python import get_Combiner
from biophysics_fitting.hay_complete_default_setup_python import (
    get_Evaluator as get_python_evaluator,
)
from tests.test_biophysics_fitting import deserialize_voltage_traces

h = neuron.h


def get_original_hay_evaluation():
    return {
        "BAC_ahpdepth": 0.14125418811813617,
        "BAC_APheight": 1.0076193107997482,
        "BAC_ISI": 0.0417182547544436,
        "BAC_caSpike_height": 0.659560205506027,
        "BAC_caSpike_width": 0.08894764532934432,
        "BAC_spikecount": 0.0,
        "bAP_spikecount": 0.0,
        "bAP_APheight": 1.7265778138894916,
        "bAP_APwidth": 2.06628632228842,
        "bAP_att2": 1.2392743531680608,
        "bAP_att3": 1.6065100454922565,
        "mf1": 1.7045454545454546,
        "AI1": 0.49272740969831885,
        "ISIcv1": 1.5149115878350792,
        "DI1": 0.42387678054719374,
        "TTFS1": 1.559524119613254,
        "APh1": 0.7270461042321147,
        "fAHPd1": 1.8012804269742928,
        "sAHPd1": 1.014632577988536,
        "sAHPt1": 0.44621364517478956,
        "APw1": 2.129551711974563,
        "mf2": 0.0,
        "AI2": 1.9222306682333696,
        "ISIcv2": 0.5017877027521275,
        "DI2": 1.4694554157331052,
        "TTFS2": 0.12556975559578443,
        "APh2": 0.9348347507146776,
        "fAHPd2": 1.4198439942151773,
        "sAHPd2": 0.3236920651878746,
        "sAHPt2": 1.1101910558238628,
        "APw2": 1.5216672210411692,
        "mf3": 2.2500225002250023,
        "AI3": 1.6812494124322965,
        "ISIcv3": 4.012157754318799,
        "DI3": 2.950213517754278,
        "TTFS3": 0.08393006742642228,
        "APh3": 0.5912669610538728,
        "fAHPd3": 1.1237604895508075,
        "sAHPd3": 0.09536464568092351,
        "sAHPt3": 2.0872162603264846,
        "APw3": 2.22822779962913,
    }


@pytest.mark.dependency(
    depends=["tests/test_biophysics_fitting/simulator_test.py::test_simulator"],
    scope="session",
)
def test_hay_evaluation_python(request, step=True):
    """Test the evaluation of the Python hay evaluator translation"""

    voltage_traces = request.config.cache.get("shared_voltage_traces", None)
    assert (
        voltage_traces is not None
    ), "The voltage traces should be set in pytest scope by now. Is the test dependency correctly configured?"
    voltage_traces = deserialize_voltage_traces(voltage_traces)

    e = get_python_evaluator(step=step)
    evaluation = e.evaluate(voltage_traces)

    # test if results are within bounds, which they should
    evaluation = get_Combiner(step=step).combine(evaluation)
    for stim, ev in evaluation.items():
        cutoff = 4.5 if "step" in stim.lower() else 3.2
        assert ev < cutoff, "Sigma cuttof of {} for {} exceeded: {}".format(
            cutoff, stim, ev
        )


@pytest.mark.dependency(
    depends=["tests/test_biophysics_fitting/simulator_test.py::test_simulator"],
    scope="session",
)
def test_hay_evaluation_python_compatibility(request):
    """test if new python translation of hay's hoc code provides the same results"""
    v = request.config.cache.get("shared_voltage_traces", None)

    evaluation_python_translation = get_python_evaluator(
        step=False, interpolate_voltage_trace=False
    ).evaluate(v)

    evaluation_hay = get_hay_Evaluator(
        step=False, interpolate_voltage_trace=False
    ).evaluate(v)

    diff = {
        k: evaluation_python_translation[k] - evaluation_hay[k] for k in evaluation_hay
    }
    rel_diff = {k: diff[k] / evaluation_hay[k] for k in diff.keys()}
    for ev, rd in rel_diff.items():
        assert (
            np.abs(rd) < 1e-10
        ), "{stim} is off by more than just a numerical truncation factor: {rd}".format(
            stim=ev, rd=rd
        )


@pytest.mark.skip("This test segfaults, unsure why. It's set for removal either way.")
def test_hay_evaluation():
    """compare the result of the optimization of the hay evaluator with a precomputed result"""
    print(
        "Testing this only works, if you uncomment the following line in MOEA_gui_for_objective_calculation.hoc: "
    )
    print('// CreateNeuron(cell,"GAcell_v3") remove comment ")')
    print(
        "However, this will slow down every NEURON evaluation (as an additional cell is created which will be"
    )
    print(
        "Included in all simulation runs. Therefore change this such that the cell is deleted afterwards or "
    )
    print("comment out the line again.")

    x = get_feasible_model_params().x
    y_new = hay_objective_function(x)
    y = get_feasible_model_objectives().y
    try:
        assert max(np.abs((y - y_new[y.index].values))) < 0.05
    except:
        print(y)
        print(y_new[y.index].values)
        raise


if __name__ == "__main__":
    test_hay_evaluation_python_compatibility()
