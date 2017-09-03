import Interface as I
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

class ReducedModelKernel(dict):
    pass

def _kernel_postprocess(clfs, n = 2, names = ['EXC', 'INH']):
    '''splits kernel which first contains EXC afterwards INH values in half.
    Returns: dictionary with keys EXC and INH
    
    This function is not pure!'''
    clfs['kernel_dict'] = []
    for clf in clfs['classifier_']:
        l = len(clf.coef_[0])
        assert(l%n == 0) ### even numer
        ln = l/n
        out = ReducedModelKernel()
        for lv in range(n):  
            name = names[lv]
            interval = clfs['classifier_'][0].coef_[0][lv*ln:(lv+1)*ln]
            out[name] = interval
        clfs['kernel_dict'].append(out)

def get_kernel_C2_grid(output_window_min = 255, output_window_max = 265, input_window_width = 80, normalize_group_size = True, plot = True):
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
    
    if plot: 
        lda_values = np.dot(np.concatenate([X[:,180:265], X[:,180+300:265+300]], axis = 1), clfs['classifier_'][0].coef_[0])
        pdf = pd.DataFrame(dict(lda_values = lda_values, spike = y))
        lookup_series = pdf.groupby(pdf.lda_values.round()).apply(lambda x: len(x[x.spike])/float(len(x)))
        fig = plt.figure(figsize = (15,3), dpi = 300)
        lookup_series.plot(ax = fig.add_subplot(111))
        fig.axes[-1].set_title('probability of spike in interval [255:265]')
        pdf.groupby(pdf.lda_values.round()).apply(lambda x: len(x)).plot(secondary_y = True)

    return _kernel_postprocess(clfs), lookup_series, pdf

def C2_grid_10ms_to_20ms_poststim():
    '''returns : clfs, lookup_series, pdf'''
    warnings.warn(DeprecationWarning("Deprecated: Use get_kernel_C2_grid instead"))
    return get_kernel_C2_grid(output_window_min = 255, output_window_max = 265)
    