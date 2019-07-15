import cPickle as pickle
import Interface as I
import sklearn.metrics
import scipy.optimize

# def instancemethod_serialization_helper(pickled_instance, method_name, 
#                                          method_args=[], method_kwargs = {}):
#     instance_ = pickle.loads(pickled_instance)
#     method_ = getattr(instance_, method_name)
#     return method_(*method_args, **method_kwargs)

def solver_serialization_helper(pickled_solver, split, x0):
    solver = pickle.loads(pickled_solver)
    solver.strategy.set_split(split)
    return solver.optimize(x0 = x0)
class Rm(object):
    def __init__(self, name, mdb, tmin = None, tmax = None, width = None):
        self.name = name
        self.mdb = mdb
        self.tmax = tmax
        self.tmin = tmin
        self.width = width
        self.n_trials = None 
        self.data_extractors = {}
        self.strategies = {}
        self.Data = DataView()
        self.Data.setup(self)
        self.DataSplitEvaluation = DataSplitEvaluation(self)
        # for remote optimization
        self.results_remote = False # flag, false if we have all results locally
    
    def add_data_extractor(self, name, data_extractor, setup = True):
        self.data_extractors[name] = data_extractor
        if setup == True:
            data_extractor.setup(self)
    
    def add_strategy(self, strategy, setup = True, view = None):
        name = strategy.name
        assert(name not in self.strategies.keys())
        self.strategies[name] = strategy
        if view is None:
            view = self.Data
        if setup:
            strategy.setup(view, self.DataSplitEvaluation)
            
    def get_n_trials(self):
        if self.n_trials is None:
            self.n_trials = len(self.Data['y'])
        return self.n_trials
        
    def extract(self, name):
        return self.data_extractors[name].get()
    
    def run(self, client = None):
        for strategy_name in sorted(self.strategies.keys()):
            strategy = self.strategies[strategy_name]
            for solver_name in sorted(strategy.solvers.keys()):
                solver = strategy.solvers[solver_name]
                if client is not None:
                    print 'starting remote optimization', strategy_name, solver_name
                    solver.optimize_all_splits(client)
                    self.results_remote = True
                else:
                    print 'starting local optimization', strategy_name, solver_name
                    solver.optimize_all_splits()
    
    def _gather_results(self, client):
        assert(client is not None)
        self.DataSplitEvaluation.optimizer_results = \
            client.gather(self.DataSplitEvaluation.optimizer_results)
        self.results_remote = False
        
    def get_results(self, client = None):
        if self.results_remote:
            self._gather_results(client)
        return self.DataSplitEvaluation.compute_scores()            
class Strategy(object):
    def __init__(self, name):
        self.name = name
        self.solvers = {}
        self.split = None
        
    def setup(self, data, DataSplitEvaluation):
        self.data = data
        self.DataSplitEvaluation = DataSplitEvaluation
        self._setup()
        
    def _setup(self):
        pass
    
    def _get_x0(self):
        pass
    
    def set_split(self, split):
        self.split = split
        return self
    
    def get_score(self, x):
        score = self._get_score(x)        
        if self.split:
            return score[self.split]
        else:
            return score
        
    def get_y(self):
        if self.split:
            return self.data['y'][self.split]
        else:
            return self.data['y']
    
    def _objective_function(self, x):
        s = self.get_score(x)
        y = self.get_y()
        return -1 * sklearn.metrics.roc_auc_score(y, s) 
    
    def add_solver(self, solver, setup = True):
        assert solver.name not in self.solvers.keys()
        self.solvers[solver.name] = solver
        if setup:
            solver.setup(self)
class Solver(object):
    def __init__(self, name):
        self.name = name
        
    def setup(self, strategy):
        self.strategy = strategy
    
    def optimize_all_splits(self, client = None):
        out = {}
        if client:
            self_serialized = pickle.dumps(self)
            self_serialized = client.scatter(self_serialized)
        for name, split in self.strategy.DataSplitEvaluation.splits.iteritems():
            if client:
                x0 = self.strategy._get_x0()
                out[name] = client.submit(solver_serialization_helper,
                                          self_serialized, #pickle.dumps(self),
                                          split['train'],
                                          x0)
                # out[name] = client.submit(self.optimize, x0 = x0)
            else:
                self.strategy.set_split(split['train'])
                out[name] = self.optimize()
        self.strategy.DataSplitEvaluation.add_result(self, out)
        return out
class DataExtractor(object):
    def get(self):
        pass
    
    def setup(self, Rm):
        pass 
class DataView(object):
    def __init__(self, mapping_dict = {}):
        self.mapping_dict = mapping_dict
        
    def setup(self, Rm):
        self.Rm = Rm
        
    def __getitem__(self, key):
        if not key in self.mapping_dict:
            return self.Rm.extract(key)
        else:
            return self.Rm.extract(self.mapping_dict[key])
class DataSplitEvaluation(object):
    '''class used to split data in training/test sets and 
    evaluating performance scores corresponding to the splits'''
    
    def __init__(self, Rm):
        self.Rm = Rm
        self.splits = {}
        self.solvers = []
        self.optimizer_results = []
        self.optimizer_results_keys = []
        self.scores = []
        self.scores_keys = []
        
    def add_random_split(self, name, percentage_train = .7, l = None):
        assert(name not in self.splits.keys())
        if l is None:
            n = self.Rm.get_n_trials()
            l = range(n)
        else:
            n = len(l)
        I.np.random.shuffle(l)
        train = l[:int(n*percentage_train)]
        test = l[int(n*percentage_train):]
        subtest1 = test[:int(len(test)/2)]
        subtest2 = test[int(len(test)/2):]
        self.splits[name] = {'train': train, 'test': test,
                             'subtest1': subtest1, 'subtest2': subtest2}
    
    def add_isi_dependent_random_split(self, name, min_isi = 10, percentage_train = .7):
        assert(name not in self.splits.keys())
        ISI = self.Rm.extract('ISI') * -1
        ISI = ISI.fillna(min_isi+1)
        ISI = ISI.reset_index(drop = True)
        l = list(ISI[ISI >= min_isi].index)
        self.add_random_split(name, percentage_train = percentage_train, l = l)

    def get_splits(self):
        return self.splits
    
    def add_result(self, solver, x):
        assert len(self.splits) == len(x)
        # assert solver.strategy.Rm is self.Rm        
        solver_name = solver.name
        strategy_name = solver.strategy.name
        run = len([k for k in self.optimizer_results_keys 
                   if k[0] == strategy_name and k[1] == solver_name])
        self.optimizer_results_keys.append((strategy_name, solver_name, run))        
        self.optimizer_results.append(x)
        self.solvers.append(solver)
        
    def compute_scores(self):
        strategy_index = []
        solver_index = []
        split_index = []
        subsplit_index = []
        success_index = []
        score_index = []
        x_index = []
        runs_index = []
        for k, solver, x in zip(self.optimizer_results_keys, self.solvers, self.optimizer_results):
            for split_name, xx in x.iteritems():
                split = self.splits[split_name]
                for subsplit_name, subsplit in split.iteritems():
                    runs_index.append(k[2])
                    x_index.append(xx.x)
                    success_index.append(xx.success)
                    solver_index.append(solver.name)
                    strategy = solver.strategy
                    strategy_index.append(strategy.name)
                    split_index.append(split_name)
                    subsplit_index.append(subsplit_name)
                    score = strategy.set_split(subsplit)._objective_function(xx.x)
                    score_index.append(score)
        out =  {'strategy': strategy_index,
                'solver': solver_index,
                'split': split_index,
                'subsplit': subsplit_index,
                'run': runs_index,
                'score': score_index,
                'x': x_index,
                'success': success_index}
        out = I.pd.DataFrame(out)
        out = out.set_index(['strategy','solver','split','subsplit', 'run']).sort_index()
        return out  
class RaisedCosineBasis(object):
    def __init__(self, a = 2, c = 1, phis = I.np.arange(1,11, 0.5), width = 80, reversed_ = False):
        self.a = a
        self.c = c
        self.phis = phis
        self.reversed_ = reversed_
        self.compute(width)
        
    def compute(self, width = 80):
        self.width = width
        self.t = I.np.arange(width)
        rev = -1 if self.reversed_ else 1
        self.basis = [self.get_raised_cosine(self.a,self.c,phi,self.t)[1][::rev] for phi in self.phis]
        return self
    
    def get(self):
        return self.basis
    
    def get_superposition(self, x):
        return sum([b*xx for b,xx in zip(self.basis, x)])

    def visualize(self, ax = None, plot_kwargs = {}):
        if ax is None:
            ax = I.plt.figure().add_subplot(111)
        for b in self.get():
            ax.plot(self.t, b, **plot_kwargs)
            
    def visualize_x(self, x, ax = None, plot_kwargs = {}):
        if ax is None:
            ax = I.plt.figure().add_subplot(111)        
        ax.plot(self.t, self.get_superposition(x), **plot_kwargs)
    
    @staticmethod
    def get_raised_cosine(a = 1, c = 1, phi = 0, t = I.np.arange(0,80, 1)):
        cos_arg = a*I.np.log(t+c) - phi
        v = .5*I.np.cos(cos_arg) + .5
        v[cos_arg >= I.np.pi] = 0
        v[cos_arg <= -I.np.pi] = 0
        return t,v

## data extractors
class DataExtractor_spatiotemporalSynapseActivation(DataExtractor):
    '''extracts array of the shape (trial, time, space) from spatiotemporal synapse activation binning'''
    def __init__(self, key):
        self.key = key
        
    def setup(self, Rm):
        self.mdb = Rm.mdb
        self.tmin = Rm.tmin
        self.tmax = Rm.tmax
        self.width = Rm.width
    
    @staticmethod
    def get_spatial_bin_level(key):
        '''returns the index that relects the spatial dimension'''
        return key[-1].split('__').index('binned_somadist')

    def get_spatial_binsize(self):
        '''returns spatial binsize'''
        mdb = self.mdb
        key = self.key
        level = self.get_spatial_bin_level(key)
        spatial_binsize = mdb[key].keys()[0][level] # something like '100to150'
        spatial_binsize = spatial_binsize.split('to')
        spatial_binsize = float(spatial_binsize[1])-float(spatial_binsize[0])
        return spatial_binsize

    def get_groups(self):
        '''returns all groups other than spatial binning'''
        mdb = self.mdb
        key = self.key
        level = self.get_spatial_bin_level(key)
        out = []
        for k in mdb[key].keys():
            k = list(k)
            k.pop(level)
            out.append(tuple(k))
        return set(out)

    def get_sorted_keys_by_group(self, group):
        '''returns keys sorted such that the first key is the closest to the soma'''
        mdb = self.mdb
        key = self.key        
        group = list(group)
        level = self.get_spatial_bin_level(key)
        keys = mdb[key].keys()
        keys = sorted(keys, key = lambda x: float(x[level].split('to')[0]))
        out = []
        for k in keys:
            k_copy = list(k[:])
            k_copy.pop(level)
            if k_copy == group:
                out.append(k)
        return out

    def get_spatiotemporal_input(self, group):
        '''returns spatiotemporal input in the following dimensions:
        (trial, time, space)'''
        mdb = self.mdb
        key = self.key        
        keys = self.get_sorted_keys_by_group(group)
        out = [mdb[key][k][:,self.tmax-self.width:self.tmax] for k in keys]
        out = I.np.dstack(out)
        print out.shape
        return out
    
    def get(self):
        '''returns dictionary with groups as keys and spatiotemporal inputpatterns as keys.
        E.g. if the synapse activation is grouped by excitatory / inhibitory identity and spatial bins,
        the dictionary would be {'EXC': matrix_with_dimensions[trial, time, space], 
                                 'INH': matrix_with_dimensions[trial, time, space]}'''
        return {g: self.get_spatiotemporal_input(g) for g in self.get_groups()}
class DataExtractor_spiketimes(DataExtractor):
    def setup(self, Rm):
        self.mdb = Rm.mdb
        
    def get(self):
        return self.mdb['spike_times']
class DataExtractor_spikeInInterval(DataExtractor):
    def __init__(self, tmin = None, tmax = None):
        self.tmin = tmin
        self.tmax = tmax
        
    def setup(self, Rm):
        if self.tmin is None:
            self.tmin = Rm.tmin
        if self.tmax is None:
            self.tmax = Rm.tmax
        self.mdb = Rm.mdb
            
    def get(self):
        st = self.mdb['spike_times']
        return I.spike_in_interval(st, tmin = self.tmin, tmax = self.tmax)
class DataExtractor_ISI(DataExtractor):
    def __init__(self, t = None):
        self.t = t
    
    def setup(self, Rm):
        self.mdb = Rm.mdb
        if self.t is None:
            self.t = Rm.tmin
        
    def get(self):
        st = self.mdb['spike_times']
        t = self.t
        st[st>t] = I.np.NaN
        return st.max(axis=1)
class DataExtractor_daskDataframeColumn(DataExtractor):
    def __init__(self, key, column, client = None):
        if not isinstance(key, tuple):
            self.key = (key,)
        else:
            self.key = key
        self.column = column
        self.client = client
        self.data = None
    
    def setup(self, Rm):
        self.mdb = Rm.mdb
        cache = self.mdb.create_sub_mdb('DataExtractor_daskDataframeColumn_cache', raise_ = False)
        complete_key = list(self.key) + [self.column]
        complete_key = map(str, complete_key)
        complete_key = tuple(complete_key)
        print complete_key
        if not complete_key in cache.keys():
            slice_ = self.mdb[self.key][self.column]
            slice_ = self.client.compute(slice_).result()
            cache.setitem(complete_key, slice_, dumper = I.dumper_pandas_to_msgpack)
        self.data = cache[complete_key]
        # after the setup, the object must be serializable and therefore must not contain a client objectz            
        self.client = None 
    
    def get(self):
        return self.data
class Solver_COBYLA(Solver):
    def __init__(self, name, method = 'COBYLA'):
        self.name = name        
        self.method = method
            
    def optimize(self, maxiter = 5000, x0 = None):
        if x0 is None:
            x0 = self.strategy._get_x0()
        out = scipy.optimize.minimize(self.strategy._objective_function, x0,
                                      method = self.method, options = dict(maxiter = maxiter,
                                                                       disp = True))        
        return out 
class Strategy_ISIcutoff(Strategy):
    def __init__(self, name, cutoff_range = (0,4), penalty = -10**10):
        super(Strategy_ISIcutoff, self).__init__(name)
        self.cutoff_range = cutoff_range
        self.penalty = penalty

    
    def _setup(self):
        self.ISI = self.data['ISI'].fillna(-100)
        
    def _get_score(self, x):
        x = x[0]*-1
        ISI = I.np.array(self.ISI)
        non_refractory = (ISI <= x)
        refractory = (ISI > x)
        ISI[refractory] = self.penalty
        ISI[non_refractory] = 0
        return ISI
        
    def _get_x0(self):
        min_ = self.cutoff_range[0]
        max_ = self.cutoff_range[1]
        return I.np.random.rand(1)*(max_ - min_) + min_
class Strategy_ISIexponential(Strategy):
    def __init__(self, name, max_isi = 100):
        super(Strategy_ISIexponential, self).__init__(name)
        self.name = name
        self.max_isi = 100
        
    def _setup(self):
        self.ISI = self.data['ISI']
        
    def _get_x0(self):
        return (I.np.random.rand(2)*I.np.array([-10, 15]))
    
    def _get_score(self, x):
        ISI = self.ISI
        ISI = ISI * -1
        ISI = ISI.fillna(self.max_isi)
        ISI = -1 * I.np.exp(x[0]*(ISI-x[1]))        
        return ISI.replace([I.np.inf, -I.np.inf], I.np.nan).fillna(-10**10).values
    
    def visualize(self, optimizer_output, normalize = True):
        fig = I.plt.figure()
        ax = fig.add_subplot(111)
        x = I.np.arange(0,50)
        for o in optimizer_output:
            v = -1 * I.np.exp(o.x[0]*(x-x[1]))
            if normalize:
                v = v / I.np.max(I.np.abs(v))
            ax.plot(v)
class Strategy_ISIraisedCosine(Strategy):
    def __init__(self, name, RaisedCosineBasis_postspike):
        super(Strategy_ISIraisedCosine, self).__init__(name)
        self.RaisedCosineBasis_postspike = RaisedCosineBasis_postspike
        
    def _setup(self):
        self.ISI = self.data['ISI']
        
    def _get_x0(self):
        return (I.np.random.rand(len(self.RaisedCosineBasis_postspike.phis))*2-1)*5
    
    def _get_score(self, x):
        ISI = self.ISI
        ISI = ISI * -1
        width = self.RaisedCosineBasis_postspike.width
        ISI[ISI>=width] = width
        ISI = ISI.fillna(width)
        ISI = ISI.astype(int)-1
        kernel = self.RaisedCosineBasis_postspike.get_superposition(x)
        return kernel[ISI.values]
    
    def visualize(self, optimizer_output, normalize = True, only_succesful = True):
        fig = I.plt.figure()
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
        v = v-[v[-1]]
        v = v / I.np.abs(v[0])
        return v
class Strategy_spatiotemporalRaisedCosine(Strategy):
    '''requires keys: spatiotemporalSa, st, y, ISI'''
    def __init__(self, name, RaisedCosineBasis_spatial, RaisedCosineBasis_temporal):
        super(Strategy_spatiotemporalRaisedCosine, self).__init__(name)
        self.RaisedCosineBasis_spatial = RaisedCosineBasis_spatial
        self.RaisedCosineBasis_temporal = RaisedCosineBasis_temporal

    def _setup(self):
        self.compute_basis()
        self.groups = sorted(self.base_vectors_arrays_dict.keys())
        self.len_groups = len(self.groups)
        self.len_z, self.len_t, self.len_trials = self.base_vectors_arrays_dict.values()[0].shape
        
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
                tSa = I.np.dot(stSa,z).squeeze()
                for t in self.RaisedCosineBasis_temporal.compute(len_t).get():
                    base_vector_row.append(I.np.dot(tSa, t))
                base_vector_array.append(base_vector_row)
            base_vectors_arrays_dict[group] = I.np.array(base_vector_array)
        self.base_vectors_arrays_dict = base_vectors_arrays_dict
            
    def _get_x0(self):
        return I.np.random.rand((self.len_z + self.len_t)*self.len_groups)*2-1
    
    def convert_x(self, x):
        out = {}
        x = x.reshape(self.len_groups, len(x) / self.len_groups)
        for lv, group in enumerate(self.groups):
            x_z = x[lv, :self.len_z]
            x_t = x[lv, self.len_z:]
            out[group] = x_z, x_t
        return out    
    
    def _get_score(self, x):
        outs = []
        for group, (x_z, x_t) in self.convert_x(x).iteritems():
            array = self.base_vectors_arrays_dict[group]
            out = I.np.dot(x_t, array).squeeze()
            out = I.np.dot(x_z, out).squeeze()
            outs.append(out)
        return sum(outs)
    
    def normalize(self, x, flipkey = None):
        '''normalize such that exc and inh peak is at 1 and -1, respectively.
        normalize, such that sum of all absolute values of all kernels is 1'''
        x = self.convert_x(x)
        #temporal
        b = self.RaisedCosineBasis_temporal
        x_exc_t = x[('EXC',)][1]
        x_inh_t = x[('INH',)][1]
        x_exc_z = x[('EXC',)][0]
        x_inh_z = x[('INH',)][0]
        norm_exc = b.get_superposition(x_exc_t)[I.np.argmax(I.np.abs(b.get_superposition(x_exc_t)))]
        norm_inh = -1*b.get_superposition(x_inh_t)[I.np.argmax(I.np.abs(b.get_superposition(x_inh_t)))]
        # spatial
        b = self.RaisedCosineBasis_spatial
        # norm_spatial = sum(I.np.abs(b.get_superposition(x_exc_z)) + I.np.abs(b.get_superposition(x_inh_z)))
        norm_spatial = max(I.np.abs(b.get_superposition(x_exc_z * norm_exc)))
        # print norm_exc, norm_inh, norm_spatial
        x[('EXC',)] = (x_exc_z*norm_exc/norm_spatial, x_exc_t/norm_exc)
        x[('INH',)] = (x_inh_z*norm_inh/norm_spatial, x_inh_t/norm_inh)
        # output
        x_out = []
        for group in self.groups:
            x_out += list(x[group][0]) + list(x[group][1])
        return I.np.array(x_out)
    
    def get_color_by_group(self, group):
        if 'EXC' in group:
            return 'r'
        elif 'INH' in group:
            return 'grey'
        else:
            return None

    def visualize(self, optimizer_output, only_successful = False, normalize = True):
        fig = I.plt.figure(figsize = (10,5))
        ax_z = fig.add_subplot(1,2,1)
        ax_t = fig.add_subplot(1,2,2)
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
                self.RaisedCosineBasis_temporal.visualize_x(x_t, ax = ax_t, plot_kwargs = {'c': c})
                self.RaisedCosineBasis_spatial.visualize_x(x_z, ax = ax_z, plot_kwargs = {'c': c})  
class Strategy_linearCombinationOfData(Strategy):
    def __init__(self, name, data_keys):
        super(Strategy_linearCombinationOfData, self).__init__(name)        
        self.data_keys = data_keys
        self.data_values = None
    
    def _setup(self):
        self.data_values = I.np.array([self.data[k] for k in self.data_keys])
    
    def _get_x0(self):
        return I.np.random.rand(len(self.data_keys))*2-1
    
    def _get_score(self, x):
        return I.np.dot(self.data_values.T, x)
class CombineStrategies_sum(Strategy):
    def __init__(self, name):
        super(CombineStrategies_sum, self).__init__(name)
        self.strategies = []
        self.lens = []
        self.split = None
        
    def setup(self, data, DataSplitEvaluation):
        self.data = data
        self.DataSplitEvaluation = DataSplitEvaluation
        for s in self.strategies:
            s.setup(data,DataSplitEvaluation)
            self.lens.append(len(s._get_x0()))
            
    def set_split(self, split):
        self.split = split
        for s in self.strategies:
            print 'Setting split in ', s.name
            s.set_split(split)
        return self
        
    def add_strategy(self, s, setup = True):
        self.strategies.append(s)
    
    def get_score(self, x):
        out = 0
        len_ = 0
        for s, l in zip(self.strategies, self.lens):
            out += s.get_score(x[len_:len_+l])
            len_ += l
        return out
    
    def _get_x0(self):
        out = [s._get_x0() for s in self.strategies]
        return I.np.concatenate(out)                