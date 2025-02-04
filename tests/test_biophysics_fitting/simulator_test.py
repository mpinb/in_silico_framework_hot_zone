"Test Simulator on morphology 89"

import os

import pandas as pd
import pytest

from tests.test_biophysics_fitting import TEST_DATA_FOLDER, get_test_simulator_morph89, serialize_voltage_traces


def get_example_models_89():
    example_models = pd.read_csv(
        os.path.join(TEST_DATA_FOLDER, "neuron_models_morph_89.csv")
    )
    biophysical_parameters = [
        e for e in example_models.columns if "ephys" in e or e == "scale_apical.scale"
    ]
    return example_models[biophysical_parameters]


@pytest.mark.dependency(scope="session")
def test_simulator(request, step=True):
    s = get_test_simulator_morph89(step=step)
    p = get_example_models_89().iloc[0]
    v = s.run(p)

    # make json serializable
    request.config.cache.set('shared_voltage_traces', serialize_voltage_traces(v))

if __name__ == "__main__":
    test_simulator(None)
