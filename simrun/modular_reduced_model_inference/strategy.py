import numpy as np
from functools import partial
import sklearn.metrics
import matplotlib.pyplot as plt
import weakref
import sys
from config.isf_logging import logger
CUPY_ENABLED = 'cupy' in sys.modules
if CUPY_ENABLED:
    import cupy
    np = cupy
else:
    logger.warning("CUPY is not available.")
    import numpy as np

# cupy frees GPU memory when all references are deleted
# as this is difficult to track, use the make_weakref method, which stores all
# GPU arrays in the _WEAKREF_ARRAY_LIST and returns a weakref object. This can be used to
# interact with the data, but is not a reference.
# Therefore, it is sufficient to empty _WEAKREF_ARRAY_LIST, which frees the GPU memory.
# All weakref objects pointing to GPU arrays will then be invalidated.
_WEAKREF_ARRAY_LIST = []

def make_weakref(obj):
    """Create a weak reference of a Python object.

    Objects saved on VRAM do not get cleared automatically,
    and memory management needs to be done manually.
    However, weak references get cleared by the Python garbage collector.
    This is a convenience method to convert Python objects to weak references,
    so that memory handling is more robust and practical.

    Attention:
        A weak reference is not the same as a direct reference.
        Objects with references to it (i.e; referents) do not get destroyed as long as a direct reference exists.
        This is not the case for weak references.
    """    
    _WEAKREF_ARRAY_LIST.append(obj)
    return weakref.proxy(obj)


def dereference(weakrefobj):
    '''Dereference a reference and fetch the referent.

    Attention:
        Uses private interface ... check after version update!

    See also:
        https://stackoverflow.com/questions/19621036/acquiring-a-regular-reference-from-a-weakref-proxy-in-python

    Args:
        weakrefobj (wearkef.proxy.object): The weak reference to an object.

    Returns:
        obj: The underlying referent object referred to by the wear reference.

    '''
    return weakrefobj.__repr__.__self__


def clear_memory():
    """Remove all weak references
    
    Cupy frees GPU memory when all references are deleted
    As this is difficult to track, use the :py:meth:`simrun.modular_reduced_model_inference.make_weakref` method, which storesall GPU arrays in the _WEAKREF_ARRAY_LIST and returns a weakref object. 
    This can be used to interact with the data, but is not a reference.
    Therefore, it is sufficient to empty _WEAKREF_ARRAY_LIST, which frees the GPU memory.
    All weakref objects pointing to GPU arrays will then be invalidated.
    """
    del _WEAKREF_ARRAY_LIST[:]


def convert_to_numpy(x):
    """Convert a numpy to a cupy array

    Only performs this conversion if CUPY is available.

    Args:
        x (cupy.array): the array to convert.
    """
    if CUPY_ENABLED:
        return cupy.asnumpy(x)
    else:
        return x


class _Strategy(object):
    """Strategy base class.
    
    Cost function to provide to the optimizer.
    
    As a function of the parameters, compute a value for each trial.
    The optimizer will optimize for this value (highest AUROC score)
    
    Needs some repr for input data.
    
    E.G. A strategy that needs to optimize for AP refractory, then the Strategy needs to incorporate this data
    """

    def __init__(self, name):
        self.name = name
        self.solvers = {}
        # self.split = None
        self.cupy_split = None
        self.numpy_split = None
        self.setup_done = False

    def setup(self, data, DataSplitEvaluation):
        if self.setup_done:
            return
        self.data = data
        self.DataSplitEvaluation = DataSplitEvaluation
        self.y = self.data['y'].values.astype('f4')
        self._setup()
        self.get_y = partial(self.get_y_static, self.y)
        self.get_score = partial(self.get_score_static, self._get_score)
        self._objective_function = partial(self._objective_function_static,self.get_score, self.get_y)
        self.setup_done = True

    def _setup(self):
        pass

    def _get_x0(self):
        pass

    def set_split(self, split, setup=True):
        cupy_split = make_weakref(np.array(split))  # cupy, if cupy is there, numpy otherwise
        numpy_split = np.array(split)  # allways numpy
        self.get_score = partial(
            self.get_score_static,
            self._get_score,
            cupy_split=cupy_split)
        self.get_y = partial(
            self.get_y_static,
            self.y,
            numpy_split=numpy_split)
        self._objective_function = partial(
            self._objective_function_static,
            self.get_score, 
            self.get_y)
        if setup:
            for solver in self.solvers.values():
                solver._setup()
        return self

    @staticmethod
    def get_score_static(_get_score, x, cupy_split=None):
        x = np.array(x).astype('f4')
        score = _get_score(x)
        #         assert len(score[dereference(cupy_split)]) < len(score)

        if cupy_split is not None:
            return score[dereference(cupy_split)]
        else:
            return score

    @staticmethod
    def get_y_static(y, numpy_split=None):
        #         assert len(y[numpy_split]) <len(y)
        if numpy_split is not None:
            return y[numpy_split]
        else:
            return y

    @staticmethod
    def _objective_function_static(get_score, get_y, x):
        s = get_score(x)
        y = get_y()
        return -1 * sklearn.metrics.roc_auc_score(y, convert_to_numpy(s))

    def add_solver(self, solver, setup=True):
        assert solver.name not in self.solvers.keys()
        self.solvers[solver.name] = solver
        if setup:
            solver.setup(self)


class Strategy_categorizedTemporalRaisedCosine(_Strategy):
    '''requires keys: spatiotemporalSa, st, y, ISI'''

    def __init__(self, name, RaisedCosineBasis_temporal):
        super(Strategy_categorizedTemporalRaisedCosine, self).__init__(name)
        self.RaisedCosineBasis_temporal = RaisedCosineBasis_temporal

    def _setup(self):
        self.compute_basis()
        self.groups = sorted(self.base_vectors_arrays_dict.keys())
        self.len_t, self.len_trials = self.base_vectors_arrays_dict.values()[0].shape
        self._get_score = partial(
            self._get_score_static,
            self.base_vectors_arrays_dict)

    def compute_basis(self):
        '''computes_base_vector_array with shape (spatial, temporal, trials)'''
        st = self.data['st']
        stSa_dict = self.data['categorizedTemporalSa']
        base_vectors_arrays_dict = {}

        for group, tSa in stSa_dict.iteritems():
            len_trials, len_t = tSa.shape
            base_vector_rows = []
            for t in self.RaisedCosineBasis_temporal.compute(len_t).get():
                base_vector_rows.append(np.dot(tSa, t))
            base_vectors_arrays_dict[group] = make_weakref(
                np.array(np.array(base_vector_rows).astype('f4')))
        self.base_vectors_arrays_dict = base_vectors_arrays_dict
        self.keys = sorted(base_vectors_arrays_dict.keys())

    def _get_x0(self):
        return np.random.rand(self.len_t * len(self.groups)) * 2 - 1

    @staticmethod
    def _get_score_static(base_vectors_arrays_dict, x):
        outs = []
        x_reshaped = x.reshape(len(base_vectors_arrays_dict), -1)
        keys = sorted(base_vectors_arrays_dict.keys())
        for lv, group in enumerate(keys):
            array = base_vectors_arrays_dict[group]
            x_current = x_reshaped[lv, :]
            out = np.dot(dereference(x_current), dereference(array)).squeeze()
            outs.append(out)
        return np.vstack(outs).sum(axis=0)


class Strategy_ISIcutoff(_Strategy):

    def __init__(self, name, cutoff_range=(0, 4), penalty=-10**10):
        super(Strategy_ISIcutoff, self).__init__(name)
        self.cutoff_range = cutoff_range
        self.penalty = penalty

    def _setup(self):
        self.ISI = make_weakref(np.array(self.data['ISI'].fillna(-100)))
        self._get_score = partial(self._get_score_static, self.ISI,self.penalty)

    @staticmethod
    def _get_score_static(ISI, penalty, x):
        """Compute the objective value for the given parameters x."""
        # hard cutoff for ISI.
        ISIc = ISI.copy()
        x = x[0] * -1
        ISIc[ISI <= x] = 0          # very good
        ISIc[ISI > x] = penalty     # very bad
        return ISIc

    def _get_x0(self):
        min_ = self.cutoff_range[0]
        max_ = self.cutoff_range[1]
        return np.random.rand(1) * (max_ - min_) + min_


class Strategy_ISIexponential(_Strategy):
    """TODO: is this fully implemented? There doesnt seem to be an actual exponential here...
    :skip-doc:
    """
    def __init__(self, name, max_isi=100):
        super(Strategy_ISIexponential, self).__init__(name)
        self.name = name
        self.max_isi = 100

    def _setup(self):
        ISI = self.data['ISI']
        ISI = ISI * -1
        ISI = ISI.fillna(self.max_isi)
        self.ISI = ISI
        self._get_score = partial(self._get_score_static, self.ISI)

    @staticmethod
    def _get_x0():
        return (np.random.rand(2) * np.array([-10, 15]))

    @staticmethod
    def _get_score_static(ISI, x):
        """NAN if no preceding AP"""
        ISI = ISI.replace([np.inf, -np.inf], np.nan).fillna(-10**10).values
        return np.array(ISI)

    def visualize(self, optimizer_output, normalize=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.arange(0, 50)
        for o in optimizer_output:
            v = -1 * np.exp(o.x[0] * (x - x[1]))
            if normalize:
                v = v / np.max(np.abs(v))
            ax.plot(v)

# TODO: This is the solver they landed on: filter out recent spikes and fit raised cosines. The other strategies
# were other ideas that didnt work
class Strategy_ISIraisedCosine(_Strategy):

    def __init__(self, name, RaisedCosineBasis_postspike):
        super(Strategy_ISIraisedCosine, self).__init__(name)
        # datatype needs to match backend, recompute
        self.RaisedCosineBasis_postspike = RaisedCosineBasis_postspike
        RaisedCosineBasis_postspike.backend = np
        RaisedCosineBasis_postspike.compute()

    def _setup(self):
        ISI = self.data['ISI']
        ISI = ISI * -1
        width = self.RaisedCosineBasis_postspike.width
        ISI[ISI >= width] = width
        ISI = ISI.fillna(width)
        ISI = ISI.astype(int) - 1
        self.ISI = make_weakref(np.array(ISI))
        self._get_score = partial(self._get_score_static, self.RaisedCosineBasis_postspike, self.ISI)

    def _get_x0(self):
        return (np.random.rand(len(self.RaisedCosineBasis_postspike.phis)) * 2 - 1) * 5

    @staticmethod
    def _get_score_static(RaisedCosineBasis_postspike, ISI, x):
        kernel = RaisedCosineBasis_postspike.get_superposition(x)
        return kernel[dereference(ISI)]

    def visualize(self, optimizer_output, normalize=True, only_succesful=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for x in optimizer_output:
            if only_succesful:
                if not x.success:
                    continue
            if normalize:
                v = self.normalize_x(x.x)
            else:
                v = self.RaisedCosineBasis_postspike.get_superposition(x.x)
            ax.plot(v)

    def normalize_x(self, x):
        v = self.RaisedCosineBasis_postspike.get_superposition(x)
        v = v - [v[-1]]
        v = v / np.abs(v[0])
        return v


class Strategy_spatiotemporalRaisedCosine(_Strategy):
    '''requires keys: spatiotemporalSa, st, y, ISI'''

    def __init__(self, name, RaisedCosineBasis_spatial,
                 RaisedCosineBasis_temporal):
        super(Strategy_spatiotemporalRaisedCosine, self).__init__(name)
        self.RaisedCosineBasis_spatial = RaisedCosineBasis_spatial
        self.RaisedCosineBasis_temporal = RaisedCosineBasis_temporal

    def _setup(self):
        self.compute_basis()
        self.groups = sorted(self.base_vectors_arrays_dict.keys())
        self.len_z, self.len_t, self.len_trials = self.base_vectors_arrays_dict.values(
        )[0].shape
        self.convert_x = partial(self._convert_x_static, self.groups,
                                   self.len_z)
        self._get_score = partial(self._get_score_static, self.convert_x,
                                    self.base_vectors_arrays_dict)

    def compute_basis(self):
        '''computes_base_vector_array with shape (spatial, temporal, trials)'''
        st = self.data['st']
        stSa_dict = self.data['spatiotemporalSa']
        base_vectors_arrays_dict = {}
        for group, stSa in stSa_dict.iteritems():
            len_trials, len_t, len_z = stSa.shape
            base_vector_array = []
            for z in self.RaisedCosineBasis_spatial.compute(len_z).get():
                base_vector_row = []
                tSa = np.dot(stSa, z).squeeze()
                for t in self.RaisedCosineBasis_temporal.compute(len_t).get():
                    base_vector_row.append(np.dot(tSa, t))
                base_vector_array.append(base_vector_row)
            base_vectors_arrays_dict[group] = make_weakref(
                np.array(np.array(base_vector_array).astype('f4')))
        self.base_vectors_arrays_dict = base_vectors_arrays_dict

    def _get_x0(self):
        return np.random.rand(
            (self.len_z + self.len_t) * len(self.groups)) * 2 - 1

    @staticmethod
    def _convert_x_static(groups, len_z, x):
        len_groups = len(groups)
        out = {}
        x = x.reshape(len_groups, len(x) / len_groups)
        for lv, group in enumerate(groups):
            x_z = x[lv, :len_z]
            x_t = x[lv, len_z:]
            out[group] = x_z, x_t
        return out

    @staticmethod
    def _get_score_static(convert_x, base_vectors_arrays_dict, x):
        outs = []
        for group, (x_z, x_t) in convert_x(x).iteritems():
            array = base_vectors_arrays_dict[group]
            out = np.dot(dereference(x_t), dereference(array)).squeeze()
            out = np.dot(dereference(x_z), dereference(out)).squeeze()
            outs.append(out)
        return np.vstack(outs).sum(axis=0)

    def normalize(self, x, flipkey=None):
        '''normalize such that exc and inh peak is at 1 and -1, respectively.
        normalize, such that sum of all absolute values of all kernels is 1'''
        x = self.convert_x(x)
        #temporal
        b = self.RaisedCosineBasis_temporal
        x_exc_t = x[('EXC',)][1]
        x_inh_t = x[('INH',)][1]
        x_exc_z = x[('EXC',)][0]
        x_inh_z = x[('INH',)][0]
        norm_exc = b.get_superposition(x_exc_t)[np.argmax(
            np.abs(b.get_superposition(x_exc_t)))]
        norm_inh = -1 * b.get_superposition(x_inh_t)[np.argmax(
            np.abs(b.get_superposition(x_inh_t)))]
        # spatial
        b = self.RaisedCosineBasis_spatial
        # norm_spatial = sum(np.abs(b.get_superposition(x_exc_z)) + np.abs(b.get_superposition(x_inh_z)))
        norm_spatial = max(np.abs(b.get_superposition(x_exc_z * norm_exc)))
        # print norm_exc, norm_inh, norm_spatial
        x[('EXC',)] = (x_exc_z * norm_exc / norm_spatial, x_exc_t / norm_exc)
        x[('INH',)] = (x_inh_z * norm_inh / norm_spatial, x_inh_t / norm_inh)
        # output
        x_out = []
        for group in self.groups:
            x_out += list(x[group][0]) + list(x[group][1])
        return np.array(x_out)

    def get_color_by_group(self, group):
        if 'EXC' in group:
            return 'r'
        elif 'INH' in group:
            return 'grey'
        else:
            return None

    def visualize(self,
                  optimizer_output,
                  only_successful=False,
                  normalize=True):
        fig = plt.figure(figsize=(10, 5))
        ax_z = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        for out in optimizer_output:
            if only_successful:
                if not out.success:
                    continue
            if normalize:
                dict_ = self.convert_x(self.normalize(out.x))
            else:
                dict_ = self.convert_x(out.x)
            for group, (x_z, x_t) in dict_.iteritems():
                c = self.get_color_by_group(group)
                self.RaisedCosineBasis_temporal.visualize_x(
                    x_t, ax=ax_t, plot_kwargs={'c': c})
                self.RaisedCosineBasis_spatial.visualize_x(x_z,
                                                           ax=ax_z,
                                                           plot_kwargs={'c': c})


class Strategy_temporalRaisedCosine_spatial_cutoff(_Strategy):
    '''requires keys: temporalSa, st, y, ISI'''

    def __init__(self, name, RaisedCosineBasis_spatial,
                 RaisedCosineBasis_temporal):
        super(Strategy_spatiotemporalRaisedCosine, self).__init__(name)
        self.RaisedCosineBasis_spatial = RaisedCosineBasis_spatial
        self.RaisedCosineBasis_temporal = RaisedCosineBasis_temporal

    def _setup(self):
        self.compute_basis()
        self.groups = sorted(self.base_vectors_arrays_dict.keys())
        self.len_z, self.len_t, self.len_trials = self.base_vectors_arrays_dict.values(
        )[0].shape
        self.convert_x = partial(self._convert_x_static, self.groups,
                                   self.len_z)
        self._get_score = partial(self._get_score_static, self.convert_x,
                                    self.base_vectors_arrays_dict)

    def compute_basis(self):
        '''computes_base_vector_array with shape (spatial, temporal, trials)'''
        st = self.data['st']
        stSa_dict = self.data['spatiotemporalSa']
        base_vectors_arrays_dict = {}
        for group, stSa in stSa_dict.iteritems():
            len_trials, len_t, len_z = stSa.shape
            base_vector_array = []
            for z in self.RaisedCosineBasis_spatial.compute(len_z).get():
                base_vector_row = []
                tSa = np.dot(stSa, z).squeeze()
                for t in self.RaisedCosineBasis_temporal.compute(len_t).get():
                    base_vector_row.append(np.dot(tSa, t))
                base_vector_array.append(base_vector_row)
            base_vectors_arrays_dict[group] = make_weakref(
                np.array(np.array(base_vector_array).astype('f4')))
        self.base_vectors_arrays_dict = base_vectors_arrays_dict

    def _get_x0(self):
        return np.random.rand(
            (self.len_z + self.len_t) * len(self.groups)) * 2 - 1

    @staticmethod
    def _convert_x_static(groups, len_z, x):
        len_groups = len(groups)
        out = {}
        x = x.reshape(len_groups, len(x) / len_groups)
        for lv, group in enumerate(groups):
            x_z = x[lv, :len_z]
            x_t = x[lv, len_z:]
            out[group] = x_z, x_t
        return out

    @staticmethod
    def _get_score_static(convert_x, base_vectors_arrays_dict, x):
        outs = []
        for group, (x_z, x_t) in convert_x(x).iteritems():
            array = base_vectors_arrays_dict[group]
            out = np.dot(dereference(x_t), dereference(array)).squeeze()
            out = np.dot(dereference(x_z), dereference(out)).squeeze()
            outs.append(out)
        return np.vstack(outs).sum(axis=0)

    def normalize(self, x, flipkey=None):
        '''normalize such that exc and inh peak is at 1 and -1, respectively.
        normalize, such that sum of all absolute values of all kernels is 1'''
        x = self.convert_x(x)
        #temporal
        b = self.RaisedCosineBasis_temporal
        x_exc_t = x[('EXC',)][1]
        x_inh_t = x[('INH',)][1]
        x_exc_z = x[('EXC',)][0]
        x_inh_z = x[('INH',)][0]
        norm_exc = b.get_superposition(x_exc_t)[np.argmax(
            np.abs(b.get_superposition(x_exc_t)))]
        norm_inh = -1 * b.get_superposition(x_inh_t)[np.argmax(
            np.abs(b.get_superposition(x_inh_t)))]
        # spatial
        b = self.RaisedCosineBasis_spatial
        # norm_spatial = sum(np.abs(b.get_superposition(x_exc_z)) + np.abs(b.get_superposition(x_inh_z)))
        norm_spatial = max(np.abs(b.get_superposition(x_exc_z * norm_exc)))
        # print norm_exc, norm_inh, norm_spatial
        x[('EXC',)] = (x_exc_z * norm_exc / norm_spatial, x_exc_t / norm_exc)
        x[('INH',)] = (x_inh_z * norm_inh / norm_spatial, x_inh_t / norm_inh)
        # output
        x_out = []
        for group in self.groups:
            x_out += list(x[group][0]) + list(x[group][1])
        return np.array(x_out)

    def get_color_by_group(self, group):
        if 'EXC' in group:
            return 'r'
        elif 'INH' in group:
            return 'grey'
        else:
            return None

    def visualize(self,
                  optimizer_output,
                  only_successful=False,
                  normalize=True):
        fig = plt.figure(figsize=(10, 5))
        ax_z = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        for out in optimizer_output:
            if only_successful:
                if not out.success:
                    continue
            if normalize:
                dict_ = self.convert_x(self.normalize(out.x))
            else:
                dict_ = self.convert_x(out.x)
            for group, (x_z, x_t) in dict_.iteritems():
                c = self.get_color_by_group(group)
                self.RaisedCosineBasis_temporal.visualize_x(
                    x_t, ax=ax_t, plot_kwargs={'c': c})
                self.RaisedCosineBasis_spatial.visualize_x(x_z,
                                                           ax=ax_z,
                                                           plot_kwargs={'c': c})


class Strategy_linearCombinationOfData(_Strategy):

    def __init__(self, name, data_keys):
        super(Strategy_linearCombinationOfData, self).__init__(name)
        self.data_keys = data_keys
        self.data_values = None

    def _setup(self):
        self.data_values = np.array([self.data[k] for k in self.data_keys])
        self._get_score = partial(self._get_score_static, self.data_values)

    def _get_x0(self):
        return np.random.rand(len(self.data_keys)) * 2 - 1

    @staticmethod
    def _get_score_static(data_values, x):
        return np.dot(data_values.T, x)


class CombineStrategies_sum(_Strategy):

    def __init__(self, name):
        super(CombineStrategies_sum, self).__init__(name)
        self.strategies = []
        self.lens = []
        self.split = None

    def setup(self, data, DataSplitEvaluation):
        super(CombineStrategies_sum, self).setup(data, DataSplitEvaluation)
        for s in self.strategies:
            s.setup(data, DataSplitEvaluation)
            self.lens.append(len(s._get_x0()))

    def set_split(self, split):
        super(CombineStrategies_sum, self).set_split(split)
        for s in self.strategies:
            s.set_split(split)
        return self

    def _setup(self):
        score_functions = [strategy.get_score for strategy in self.strategies]
        self._get_score = partial(
            self._get_score_static, 
            score_functions,
            self.lens)

    def add_strategy(self, s, setup=True):
        self.strategies.append(s)

    @staticmethod
    def _get_score_static(score_functions, lens, x):
        out = 0
        len_ = 0
        for sf, l in zip(score_functions, lens):
            out += sf(x[len_:len_ + l])
            len_ += l
        return out

    def _get_x0(self):
        out = [s._get_x0() for s in self.strategies]
        return np.concatenate(out)
