"Test Simulator on morphology 89"

import pytest

from tests.test_biophysics_fitting import (
    get_example_models_89,
    get_test_simulator_89,
    serialize_voltage_traces,
)


@pytest.mark.dependency(scope="session")
def test_simulator_89(request, step=True):
    """Run an ISF simulator on working biophysical parameters for morphology 89"""
    s = get_test_simulator_89(step=step)
    p = get_example_models_89().iloc[0]  # just a random working model
    v = s.run(p)

    # make json serializable
    request.config.cache.set("shared_voltage_traces", serialize_voltage_traces(v))


if __name__ == "__main__":
    test_simulator_89(None)
