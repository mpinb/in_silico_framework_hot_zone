import numpy as np
from data_base.db_initializers.prepare_ANN_batches \
    import spike_times_to_onehot
from data_base.db_initializers.load_simrun_general \
    import optimize as optimize_simrun_general
from data_base.db_initializers.synapse_activation_binning \
    import init as init_synapse_activation
from data_base.utils import silence_stdout

optimize_simrun_general = silence_stdout(optimize_simrun_general)


def test_API(fresh_db, client):
    optimize_simrun_general(fresh_db, client=client)
    init_synapse_activation(fresh_db, groupby='EI')
    init_synapse_activation(fresh_db, groupby=['EI'])
    init_synapse_activation(fresh_db, groupby=['EI', 'proximal'])


def test_onehot_encoding():
    spike_times = [
        15.999999999999625, 23.724999999999188, 30.449999999998806,
        165.79999999998614, 181.62499999997175, 298.3249999998656,
        319.37499999984647
    ]
    time_steps = [1,
                  13]  # test different time step intervals for one-hot encoding
    for time_step in time_steps:
        one_hot = spike_times_to_onehot(spike_times,
                                        min_time=0,
                                        max_time=505,
                                        time_step=time_step)
        comparison = [int(st // time_step) for st in spike_times
                     ]  # how onehot should work
        comparison = sorted(list(set(comparison)))  # remove duplicate entries
        assert all([a == b for a, b in zip(comparison,  np.where(one_hot)[0])]), \
        "One-hot encoding failed.\nSpike times: {}\none-hot coding:{}\nLocations where spike equals True:{}".format(spike_times, one_hot, np.where(one_hot)[0])
    try:
        spike_times_to_onehot([-1, -200])  # should give error
        assert False  # in case it does not give an error
    except AssertionError:
        assert True  # all good