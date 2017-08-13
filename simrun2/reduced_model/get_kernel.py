import Interface as I
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def C2_grid_10ms_to_20ms_poststim():
    '''returns : clfs, lookup_series, pdf'''
    def get_array_spike_times(mdb, min_time = 245, max_time = 285):
        st = mdb['spike_times']
        st['cell'] = st.index
        st['cell'] = st.apply(lambda x: x.cell.split('/')[0], axis = 1)
        bins = st.groupby('cell').apply(lambda x: I.temporal_binning(x, min_time = min_time, max_time = max_time, bin_size = max_time-min_time))
        location_dependent_number_of_spikes_per_trail = bins.map(lambda x: x[1][0]).to_frame('location_dependent_number_of_spikes_per_trail')
        location_dependent_number_of_spikes_per_trail_array = I.pandas_to_array(location_dependent_number_of_spikes_per_trail, \
                                 lambda index, values: int(index.split('_')[1]), \
                                 lambda index, values: int(index.split('_')[-1]), \
                                 lambda index, values: values.location_dependent_number_of_spikes_per_trail)
        return location_dependent_number_of_spikes_per_trail_array

    def get_array_from_series(s):
        cells = s.index.map(lambda x: x.split('/')[0])
        bins = s.groupby(cells).apply(lambda x: x.astype(float).mean())
        location_dependent_number_of_spikes_per_trail = bins.to_frame('location_dependent_number_of_spikes_per_trail')
        #return location_dependent_number_of_spikes_per_trail
        location_dependent_number_of_spikes_per_trail_array = I.pandas_to_array(location_dependent_number_of_spikes_per_trail, \
                                 lambda index, values: int(index.split('_')[1]), \
                                 lambda index, values: int(index.split('_')[-1]), \
                                 lambda index, values: values.location_dependent_number_of_spikes_per_trail)
        return location_dependent_number_of_spikes_per_trail_array


    def plot_array(array, fig, subplot, title = '', **kwargs):
        if isinstance(subplot, int):
            subplot = str(subplot)
            subplot = (subplot[0], subplot[1], subplot[2])
            subplot = map(int, subplot)
        sns.heatmap(array, annot = True, cbar = False, square = True, ax = fig.add_subplot(*subplot), **kwargs)
        fig.axes[-1].set_title(title)
        return fig


    def spike_in_interval(st, tmin, tmax):
        return ((st>=tmin) & (st<tmax)).any(axis = 1)    
    
    basedir = '/nas1/Data_arco/results/20170222_SuW_stimulus_in_C2_grid/'
    stim = ['B1', 'B2', 'B3', 'C1', 'C3', 'D1', 'D2', 'D3']
    mdbs = {s: I.ModelDataBase(os.path.join(basedir, s, 'mdb')) for s in stim}

    mdbs['C2'] = I.ModelDataBase('/nas1/Data_arco/results/20170214_use_cell_grid_with_soma_at_constant_depth_below_layer_4_to_evaluate_location_dependency_of_evoked_responses/mdb/')
    stim = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']

    #extract training values
    stim = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
    ys = []
    Xs = []
    sts = []
    for s in stim:
        mdb = mdbs[s]
        # target value for supervised learning
        st = mdb['spike_times']
        y = np.array(spike_in_interval(st,255,265))
        # input values
        EXC = mdb['synapse_activation', 'binned_t1', 'EI', 'proximal']['EXC', 'prox']
        INH = mdb['synapse_activation', 'binned_t1', 'EI', 'proximal']['INH', 'prox']
        X = np.concatenate([EXC, INH], axis = 1)
        #append to output variables
        ys.append(y)
        Xs.append(X)
        sts.append(st)

    y = np.concatenate(ys, axis = 0)
    X = np.concatenate(Xs, axis = 0)
    print y.shape
    print X.shape

    clfs = I.lda_prediction_rates(np.concatenate([X[:,180:265], X[:,180+300:265+300]], axis = 1),\
                                  y, verbosity = 2, return_ = 'all', normalize_group_size = True, \
                                  test_size = 0.4)

    lda_values = np.dot(np.concatenate([X[:,180:265], X[:,180+300:265+300]], axis = 1), clfs['classifier_'][0].coef_[0])
    pdf = pd.DataFrame(dict(lda_values = lda_values, spike = y))
    lookup_series = pdf.groupby(pdf.lda_values.round()).apply(lambda x: len(x[x.spike])/float(len(x)))
    fig = plt.figure(figsize = (15,3), dpi = 300)
    lookup_series.plot(ax = fig.add_subplot(111))
    fig.axes[-1].set_title('probability of spike in interval [255:265]')
    pdf.groupby(pdf.lda_values.round()).apply(lambda x: len(x)).plot(secondary_y = True)
    
    return clfs, lookup_series, pdf

	
def get_kernel_C2_grid(output_window_min = 255, output_window_max = 265, input_window_width = 80, normalize_group_size = True):
    '''returns : clfs, lookup_series, pdf'''
    def spike_in_interval(st, tmin, tmax):
        return ((st>=tmin) & (st<tmax)).any(axis = 1)    
    
    def get_mdbs_inhrobert():    
        basedir = '/nas1/Data_arco/results/20170222_SuW_stimulus_in_C2_grid/'
        stim = ['B1', 'B2', 'B3', 'C1', 'C3', 'D1', 'D2', 'D3']
        mdbs = {s: I.ModelDataBase(I.os.path.join(basedir, s, 'mdb')) for s in stim}
        mdbs['C2'] = I.ModelDataBase('/nas1/Data_arco/results/20170214_use_cell_grid_with_soma_at_constant_depth_below_layer_4_to_evaluate_location_dependency_of_evoked_responses/mdb/')
        return mdbs
    
    mdbs = get_mdbs_inhrobert()

    #extract training values
    stim = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3']
    ys = []
    Xs = []
    sts = []
    for s in stim:
        mdb = mdbs[s]
        # target value for supervised learning
        st = mdb['spike_times']
        y = I.np.array(spike_in_interval(st,output_window_min,output_window_max))
        # input values
        EXC = mdb['synapse_activation', 'binned_t1', 'EI', 'proximal']['EXC', 'prox']
        INH = mdb['synapse_activation', 'binned_t1', 'EI', 'proximal']['INH', 'prox']
        X = I.np.concatenate([EXC, INH], axis = 1)
        #append to output variables
        ys.append(y)
        Xs.append(X)
        sts.append(st)

    y = I.np.concatenate(ys, axis = 0)
    X = I.np.concatenate(Xs, axis = 0)
    print y.shape
    print X.shape

    clfs = I.lda_prediction_rates(I.np.concatenate([X[:,output_window_max-input_window_width:output_window_max], \
                                                    X[:,output_window_max-input_window_width+300:output_window_max+300]], axis = 1),\
                                  y, verbosity = 2, return_ = 'all', normalize_group_size = normalize_group_size, \
                                  test_size = 0.4)

    lda_values = I.np.dot(I.np.concatenate([X[:,output_window_max-input_window_width:output_window_max], \
                                          X[:,output_window_max-input_window_width+300:output_window_max+300]],\
                                         axis = 1), clfs['classifier_'][0].coef_[0])
    pdf = I.pd.DataFrame(dict(lda_values = lda_values, spike = y))
    lookup_series = pdf.groupby(pdf.lda_values.round()).apply(lambda x: len(x[x.spike])/float(len(x)))
    
    return clfs, lookup_series, pdf