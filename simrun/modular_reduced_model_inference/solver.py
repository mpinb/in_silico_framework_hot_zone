from functools import partial
import scipy.optimize

class Solver(object):

    def __init__(self, name):
        self.name = name

    def setup(self, strategy):
        self.strategy = strategy
        self._setup()

    def _setup(self):
        pass

    def optimize_all_splits(self, client=None, workers=None):
        out = {}
        for name, split in self.strategy.DataSplitEvaluation.splits.iteritems():
            x0 = self.strategy._get_x0()
            self.strategy.set_split(split['train'])
            if client:

                out[name] = client.submit(self.optimize, x0=x0, workers=workers)
            else:
                out[name] = self.optimize(x0=x0)
        self.strategy.DataSplitEvaluation.add_result(self, out)
        return out


    def optimize_one_split(self, client=None, workers=None, index=0):  #rieke
        out = {}
        names = sorted(self.strategy.DataSplitEvaluation.splits.keys())
        #         for name, split in self.strategy.DataSplitEvaluation.splits.iteritems():
        name = names[index]
        split = self.strategy.DataSplitEvaluation.splits[name]
        x0 = self.strategy._get_x0()
        self.strategy.set_split(split['train'])
        if client:
            out[name] = client.submit(self.optimize, x0=x0, workers=workers)
        else:
            out[name] = self.optimize(x0=x0)
        self.strategy.DataSplitEvaluation.add_result(self, out)
        return out


class Solver_COBYLA(Solver):

    def __init__(self, name):
        self.name = name

    def _setup(self):
        self.optimize = partial(
            self._optimize,
            self.strategy._objective_function,
            maxiter=5000)

    @staticmethod
    def _optimize(_objective_function, maxiter=5000, x0=None):
        """
        
        Args:
            _objective-function (callable): A Strategy._get_score function
            
        """
        out = scipy.optimize.minimize(
            _objective_function,
            x0,
            method='COBYLA',
            options=dict(maxiter=maxiter, disp=True))
        return out


