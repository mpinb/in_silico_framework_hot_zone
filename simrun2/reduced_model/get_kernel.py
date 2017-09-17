import os
import numpy as np
import pandas as pd
import dask
import matplotlib.pyplot as plt
import warnings
import collections
from model_data_base import utils as mdb_utils

#############################################
# general methods
##############################################
def concatenate_return_boundaries(values, axis = 0):
    X = np.concatenate(values, axis = axis)
    upper_bounds = np.cumsum([v.shape[axis] for v in values])
    lower_bounds = [0] + list(upper_bounds[:-1])
    boundaries = zip(lower_bounds, upper_bounds)
    return X, boundaries

def spike_in_interval(st, tmin, tmax):
    return ((st>=tmin) & (st<tmax)).any(axis = 1)   

def compare_lists_by_none_values(l1, l2):
    for x,y in zip(l1,l2):
        if x is None:
            if y is not None: return False
        if y is None:
            if x is not None: return False
    return True

##############################################
# core methods for calculating lda kernels
##############################################
def get_data_dict_from_mdb(mdb, keys):
    return {key: mdb[key] for key in keys}

def _kernel_preprocess_data(mdb_list, keys_to_synapse_activation_data, \
                            synapse_acivation_window_min, synapse_activation_window_max, \
                            output_window_min, output_window_max, refractory_period = 0):
    '''takes a dictionary containing synapse activation data. This data is then
    concatenated to one large matrix, which can be used for the lda estimator.
    
    data_dict has to have the following format:
        key: ModelDataBase - instance
        value: tuple containing keys to the synapse activation date
    
    Returns:
     - the large matrix
     - spike_times
     - a dictionary with the same keys as data_dict. Values are the respective columns
       in which this data can be found in the large matrix'''
    
    if not refractory_period == 0:
        raise NotImplementedError()
    
#     print 'synapse_acivation_window_min: ', synapse_acivation_window_min
#     print 'synapse_activation_window_max: ', synapse_activation_window_max
#     print 'output_window_min: ', output_window_min
#     print 'output_window_max: ', output_window_max
#     print 'refractory_period: ', refractory_period
    

    ys = []
    Xs = []
    for mdb in mdb_list:
        # get spike times for current mdb
        st = mdb['spike_times']
        y = np.array(spike_in_interval(st, output_window_min, output_window_max))
        # get values for current mdb
        mdb_values = [mdb[k][:, synapse_acivation_window_min:synapse_activation_window_max]\
                      for k in keys_to_synapse_activation_data]
        
        X, boundaries = concatenate_return_boundaries(mdb_values, axis = 1)
        ys.append(y)
        Xs.append(X)
    
    y = np.concatenate(ys, axis = 0)
    X = np.concatenate(Xs, axis = 0)
    return X, dict(zip(keys_to_synapse_activation_data, boundaries)), y 

def _kernel_dict_from_clfs(clfs, boundaries):
    '''splits result of lda estimator based on boundaries'''  
    kernel_dict = []
    for clf in clfs['classifier_']:
        out = dict()
        for key, b in boundaries.iteritems():
            out[key] = clf.coef_[0].squeeze()[b[0]:b[1]]
        kernel_dict.append(out) 
    return kernel_dict

def interpolate_lookup_series(lookup_series):
    stepsize = 1
    diff = max(lookup_series.index) - min(lookup_series.index)
    index = np.arange(int(min(lookup_series.index) - 0.3 * diff), \
                                                   int(max(lookup_series.index) + 0.3 * diff), \
                                                   stepsize)
    s = pd.Series([np.NaN]*len(index), index = index)
    s.iloc[0]=0
    s[lookup_series.index] = lookup_series
    return s.interpolate(method = 'linear')

def get_lookup_series_from_lda_values_spike_data(lda_values, spike, mask = None,\
                                                 lookup_series_stepsize = 3):
    '''
    
    lda_values: one-dimensional array containing lda_values of trail
    spike: one-dimensional array containing True / False
    mask: one-dimensional array containing True / False.
        If True: the respective trail will be ignored.
    '''
    pdf2 = pd.DataFrame(dict(lda_values = lda_values, spike = spike))
    if mask is not None:
        pdf = pdf2[~mask]
    else:
        pdf = pdf2
    groupby_ = (pdf.lda_values/lookup_series_stepsize).round().astype(int)
    groupby_ = groupby_*lookup_series_stepsize
    lookup_series = pdf.groupby(groupby_).apply(lambda x: len(x[x.spike])/float(len(x)))
    lookup_series = pdf.groupby(pdf.lda_values.round()).apply(lambda x: len(x[x.spike])/float(len(x)))

    return interpolate_lookup_series(lookup_series)

# def get_lookup_series_depending_on_refractory_period(refractory_period, lda_values, st, binsize_calculate = 10):
#     pdf = pd.DataFrame(dict(lda_values = lda_values, \
#                         spike = spike_in_interval(st, 260, 261), \
#                         spike_before = spike_in_interval(st, 260-refractory_period, 260)))
# 
#     pdf2 = pdf[~pdf.spike_before]
#     lookup_series = pdf2.groupby((pdf2.lda_values/binsize_calculate).round()*binsize_calculate).apply(lambda x: len(x[x.spike])/float(len(x)))
#     lookup_series = interpolate_lookup_series(lookup_series)
#     return lookup_series

class ReducedLdaModelResult():
    def __init__(self, RM, lda_dict, lda_values, p_spike):
        self.RM = RM
        self.lda_dict = lda_dict
        self.lda_values = lda_values
        self.p_spike = p_spike
        
class ReducedModel():
    pass

class ReducedLdaModel(ReducedModel):
    def __init__(self, \
                keys_to_synapse_activation_data, \
                synapse_activation_window_width = 80, \
                synapse_activation_window_min = None, \
                synapse_activation_window_max = None,\
                output_window_min = 255, output_window_max = 265, refractory_period = 0,\
                normalize_group_size = True, test_size = 0.4, verbosity = 2, \
                lookup_series_stepsize = 5, cache = True):
        
        dummy = [synapse_activation_window_min, synapse_activation_window_max, synapse_activation_window_width]
        if compare_lists_by_none_values(dummy, [1, 1, 1]):
            assert(synapse_activation_window_width == synapse_activation_window_max - synapse_activation_window_min)
        elif compare_lists_by_none_values(dummy, [None, None, 1]):
            synapse_activation_window_max = output_window_max
            synapse_activation_window_min = synapse_activation_window_max - synapse_activation_window_width
        elif compare_lists_by_none_values(dummy, [1, None, 1]):
            synapse_activation_window_max = synapse_activation_window_min + synapse_activation_window_width
        elif compare_lists_by_none_values(dummy, [1, 1, None]):
            pass
        else:
            raise ValueError("synapse_activation_window not sufficiently defined.")
        
        self.keys_to_synapse_activation_data = keys_to_synapse_activation_data
        self.synapse_activation_window_min = synapse_activation_window_min
        self.synapse_activation_window_max = synapse_activation_window_max
        self.output_window_min = output_window_min
        self.output_window_max = output_window_max
        self.refractory_period = refractory_period
        self.normalize_group_size = normalize_group_size
        self.test_size = test_size
        self.verbosity = verbosity
        self.lookup_series_stepsize = lookup_series_stepsize
        
        if cache:
            self.apply_rolling = mdb_utils.cache(self.apply_rolling)
            self.apply_static = mdb_utils.cache(self.apply_static)
    
    def fit(self, mdb_list):    
        self.mdb_list = mdb_list
        X, boundaries, y = _kernel_preprocess_data(mdb_list, \
                                self.keys_to_synapse_activation_data, \
                                self.synapse_activation_window_min, \
                                self.synapse_activation_window_max, \
                                self.output_window_min, self.output_window_max, \
                                refractory_period = self.refractory_period)
        
        self.y = y
        
        import Interface as I
        self.clfs = I.lda_prediction_rates(X, y, verbosity = self.verbosity, return_ = 'all', 
                                      normalize_group_size = self.normalize_group_size, \
                                      test_size = self.test_size)
        
        self.kernel_dict = _kernel_dict_from_clfs(self.clfs, boundaries)
        ## todo: kernel_dicts
        
        self.lda_value_dicts = []
        self.lookup_series = []
        self.lda_values = []
        for kernel_dict in self.kernel_dict:
            lda_values_dict = {}
            for k in self.keys_to_synapse_activation_data:
                b = boundaries[k]
                lda_values_dict[k] = np.dot(X[:,b[0]:b[1]], kernel_dict[k])
            lda_values = sum(lda_values_dict.values())
            lookup_series = get_lookup_series_from_lda_values_spike_data(lda_values, y, \
                                         lookup_series_stepsize = self.lookup_series_stepsize)
            self.lookup_series.append(lookup_series)
            self.lda_value_dicts.append(lda_values_dict)
            self.lda_values.append(lda_values)
    
    def plot(self):
        fig = plt.figure(figsize = (15,3), dpi = 300)
        ax = fig.add_subplot(111)    
        ax.set_title('kernel shape')
        ax.set_xlabel('# time bin')
        
        # plot kernel
        for lv in range(len(self.kernel_dict)):
            d = self.kernel_dict[lv]
            ax.set_prop_cycle(None)
            for k, v in d.iteritems():
                if lv == 0:
                    ax.plot(v, label = k)
                else:
                    ax.plot(v)
        plt.legend()
        
        # plot data distribution and probability curve 
        fig = plt.figure(figsize = (15,3), dpi = 300)
        for lv in range(len(self.kernel_dict)):
            lookup_series = self.lookup_series[lv]
            lookup_series.plot(ax = fig.add_subplot(111), color = 'b')
            fig.axes[-1].set_ylabel('p_spike')
            pd.Series(self.lda_values[0]).round().astype(int).value_counts()\
                .sort_index().plot(secondary_y = True, color = 'g')
            fig.axes[-1].set_ylabel('# datapoints')
                
            
        ax = fig.axes[-1]
        ax.set_title('probability of spike in interval '\
                                   + '[{output_window_min}:{output_window_max}] depending on lda_value'\
                                   .format(output_window_min = self.output_window_min, \
                                           output_window_max = self.output_window_max))
        ax.set_xlabel('lda_value')
       
    def apply_static(self, mdb = None, data_dict = None, model_number = 0):
        if data_dict is None:
            data_dict = get_data_dict_from_mdb(mdb, self.keys_to_synapse_activation_data)
        min_index = self.synapse_activation_window_min
        max_index = self.synapse_activation_window_max
        lda_value_dict = {k: np.dot(data_dict[k][:, min_index:max_index], \
                          self.kernel_dict[model_number][k]) for k in self.keys_to_synapse_activation_data}
        lda_values = sum(lda_value_dict.values())
        p_spike = self.lookup_series[model_number].loc[lda_values.round().astype(int)]
        return ReducedLdaModelResult(self, lda_value_dict, lda_values, p_spike)
    
    @dask.delayed
    def apply_rolling(self, mdb = None, data_dict = None):
        pass
    
    @dask.delayed
    def apply_static_param(self, ):
        pass
    
    @dask.delayed
    def apply_rolling_param(self):
        pass
###################################
# methods to get special kernels
###################################            
def get_kernel_C2_grid(keys_to_synapse_activation_data = [
        ('synapse_activation', 'binned_t1', 'EI', 'proximal','EXC', 'prox'),
        ('synapse_activation', 'binned_t1', 'EI', 'proximal','INH', 'prox')], \
                synapse_activation_window_width = 80, \
                synapse_activation_window_min = None, \
                synapse_activation_window_max = None,\
                output_window_min = 255, output_window_max = 265, refractory_period = 0,\
                normalize_group_size = True, test_size = 0.4, verbosity = 2, \
                lookup_series_stepsize = 5, cache = True):
    import Interface as I
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
    rm = ReducedLdaModel(keys_to_synapse_activation_data, \
                synapse_activation_window_width, \
                synapse_activation_window_min, \
                synapse_activation_window_max,\
                output_window_min, output_window_max, refractory_period,\
                normalize_group_size, test_size, verbosity, \
                lookup_series_stepsize, cache)
    
    rm.fit(mdbs.values())
    return rm

get_kernel_C2_grid_cached = mdb_utils.cache(get_kernel_C2_grid)

# def C2_grid_10ms_to_20ms_poststim():
#     '''returns : clfs, lookup_series, pdf'''
#     warnings.warn(DeprecationWarning("Deprecated: Use get_kernel_C2_grid instead"))
#     return get_kernel_C2_grid(output_window_min = 255, output_window_max = 265)
    