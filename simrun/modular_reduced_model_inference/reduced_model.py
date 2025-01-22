import numpy as np
import pandas as pd
from config.isf_logging import logger


def get_n_workers_per_ip(workers, n):
    '''Convenience method to get a certain amount of workers per machine

    Groups all workers by their IP, fetches :paramref:`n` workers per IP,
    and returns them as a list.

    Args:
        workers (List[dask.distributed.worker.Worker]):
            List or array of dask workers.
        n (int): Amount fo workers to fetch per machine.

    Returns:
        List[dask.distributed.worker.Worker]: List of :paramref:`n`*``n_machines`` workers.
    '''
    s = pd.Series(workers)
    return s.groupby(s.str.split(':').str[1]).apply(lambda x: x[:n]).tolist()


class Rm(object):

    def __init__(
        self,
        name,
        db,
        tmin=None,
        tmax=None,
        width=None,
        selected_indices=None):
        """

        """
        self.name = name
        self.db = db
        self.tmax = tmax
        self.tmin = tmin
        self.width = width
        self.n_trials = None
        self.data_extractors = {}
        self.strategies = {}
        self.Data = DataView()
        self.Data.setup(self)
        self.DataSplitEvaluation = DataSplitEvaluation(self)
        self.selected_indices = selected_indices  # list/nested list of integer indices for selected simulation trials
        # for remote optimization
        self.results_remote = False  # flag, false if we have all results locally

    def add_data_extractor(self, name, data_extractor, setup=True):
        self.data_extractors[name] = data_extractor
        if setup == True:
            data_extractor.setup(self)

    def add_strategy(self, strategy, setup=True, view=None):
        name = strategy.name
        assert name not in self.strategies.keys()
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

    def run(self, client=None, n_workers=None, strategy_selection=None):
        for strategy_name in sorted(self.strategies.keys()):

            # 1. Extract the strategy to apply
            if strategy_selection is not None:
                if not strategy_name in strategy_selection:
                    continue
            strategy = self.strategies[strategy_name]

            # 2. Solve for this strategy
            for solver_name in sorted(strategy.solvers.keys()):
                solver = strategy.solvers[solver_name]
                if client is not None:
                    logger.info(
                        'Starting remote optimization: strategy {} with solver {}'.format(strategy_name, solver_name))
                    workers = client.scheduler_info()['workers'].keys()
                    workers = get_n_workers_per_ip(workers, n_workers)
                    solver.optimize_all_splits(client, workers=workers)
                    self.results_remote = True
                else:
                    logger.info(
                        'Starting local optimization: strategy {} with solver {}'.format(strategy_name, solver_name))
                    solver.optimize_all_splits()

    def _gather_results(self, client):
        assert client is not None
        self.DataSplitEvaluation.optimizer_results = \
            client.gather(self.DataSplitEvaluation.optimizer_results)
        self.results_remote = False

    def get_results(self, client=None):
        if self.results_remote:
            self._gather_results(client)
        return self.DataSplitEvaluation.compute_scores()


class DataView(object):
    """Convenience wrapper class to access data.

    This wrapper class redirects data extractors based on a key mapping.
    This API is used by default in :py:class:`~simrun.modular_reduced_model_inference.Rm`
    If no mapping is provided, or a requested key does not exist in the mapping, the original key is used instead.

    Example:

        >>> data = {'a': 1, 'b': 2}
        >>> dv = DataView()
        >>> dv.setup(Rm)
        >>> dv['a']
        1
        >>> dv.mapping_dict = {'a': 'b'}
        >>> dv['a']
        2
    
    Attributes:
        mapping_dict (dict):
            Mapping between requested keys and target keys.
            Used to redirect data fetching.
        Rm (:py:class:`Rm`):
            Reduced model. Set after running :py:method:`setup`
    """

    def __init__(self, mapping_dict = None):
        """
        Args:
            mapping_dict (dict): 
                Mapping between requested keys and target keys.
                Used to redirect data fetching.
        """
        mapping_dict = mapping_dict or {}
        self.mapping_dict = mapping_dict

    def setup(self, Rm):
        """Initialize from a reduced model.

        Allow access to parent :py:class:`Rm` attributes.

        Args:
            Rm (:py:class:`Rm`): The reduced model to initialize from.
        """
        self.Rm = Rm

    def __getitem__(self, key):
        """Fetch data from key.

        If the key exists in :paramref:`mapping_dict`, return the data
        associated to the key redirect instead.

        Args:
            key (str): The key to fetch.
        """
        if not key in self.mapping_dict:
            return self.Rm.extract(key)
        else:
            return self.Rm.extract(self.mapping_dict[key])


class DataSplitEvaluation(object):
    '''class used to split data in training/test sets and 
    evaluating performance scores corresponding to the splits'''

    def __init__(self, Rm):
        """
        Args:
            Rm (:py:class:`Rm`): Reduced model. Set after running :py:method:`setup`
        """
        self.Rm = Rm
        self.splits = {}
        self.solvers = []
        self.optimizer_results = []
        self.optimizer_results_keys = []
        self.scores = []
        self.scores_keys = []

    def add_random_split(self, name, percentage_train=.7, l=None):
        assert name not in self.splits.keys()
        if l is None:
            n = self.Rm.get_n_trials()
            l = range(n)
        else:
            n = len(l)
        np.random.shuffle(np.array(l))
        train = l[:int(n * percentage_train)]
        test = l[int(n * percentage_train):]
        subtest1 = test[:int(len(test) / 2)]
        subtest2 = test[int(len(test) / 2):]
        self.splits[name] = {
            'train': train,
            'test': test,
            'subtest1': subtest1,
            'subtest2': subtest2
        }

    def add_isi_dependent_random_split(self,
                                       name,
                                       min_isi=10,
                                       percentage_train=.7):
        assert name not in self.splits.keys()
        ISI = self.Rm.extract('ISI') * -1
        ISI = ISI.fillna(min_isi + 1)
        ISI = ISI.reset_index(drop=True)
        l = list(ISI[ISI >= min_isi].index)
        self.add_random_split(name, percentage_train=percentage_train, l=l)

    def get_splits(self):
        return self.splits

    def add_result(self, solver, x):
        #         assert len(self.splits) == len(x) #rieke - want to run individual splits sometimes
        # assert solver.strategy.Rm is self.Rm
        solver_name = solver.name
        strategy_name = solver.strategy.name
        run = len([
            k for k in self.optimizer_results_keys
            if k[0] == strategy_name and k[1] == solver_name
        ])
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
        for k, solver, x in zip(self.optimizer_results_keys, self.solvers,
                                self.optimizer_results):
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
                    score = strategy.set_split(subsplit)._objective_function(
                        xx.x)
                    score_index.append(score)
        out = {
            'strategy': strategy_index,
            'solver': solver_index,
            'split': split_index,
            'subsplit': subsplit_index,
            'run': runs_index,
            'score': score_index,
            'x': x_index,
            'success': success_index
        }
        out = pd.DataFrame(out)
        out = out.set_index(['strategy', 'solver', 'split', 'subsplit',
                             'run']).sort_index()
        return out


class RaisedCosineBasis(object):

    def __init__(self,
                 a=2,
                 c=1,
                 phis=numpy.arange(1, 11, 0.5),
                 width=80,
                 reversed_=False,
                 backend=numpy):
        self.a = a
        self.c = c
        self.phis = phis
        self.reversed_ = reversed_
        self.backend = backend
        self.width = width
        self.compute(self.width)

    def compute(self, width=80):
        self.width = width
        self.t = numpy.arange(width)
        rev = -1 if self.reversed_ else 1
        self.basis = [
            make_weakref(
                self.get_raised_cosine(self.a,
                                       self.c,
                                       phi,
                                       self.t,
                                       backend=self.backend)[1][::rev])
            for phi in self.phis
        ]
        return self

    def get(self):
        return self.basis

    def get_superposition(self, x):
        return sum([b * xx for b, xx in zip(self.basis, x)])

    def visualize(self, ax=None, plot_kwargs={}):
        if ax is None:
            ax = plt.figure().add_subplot(111)
        for b in self.get():
            ax.plot(self.t, b, **plot_kwargs)

    def visualize_x(self, x, ax=None, plot_kwargs={}):
        if ax is None:
            ax = plt.figure().add_subplot(111)
        ax.plot(self.t, self.get_superposition(x), **plot_kwargs)

    @staticmethod
    def get_raised_cosine(a=1,
                          c=1,
                          phi=0,
                          t=numpy.arange(0, 80, 1),
                          backend=numpy):
        cos_arg = a * numpy.log(t + c) - phi
        v = .5 * numpy.cos(cos_arg) + .5
        v[cos_arg >= numpy.pi] = 0
        v[cos_arg <= -numpy.pi] = 0
        return backend.array(t.astype('f4')), backend.array(v.astype('f4'))
