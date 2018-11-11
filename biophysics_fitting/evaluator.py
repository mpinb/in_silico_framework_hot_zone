'''
Created on Nov 08, 2018

@author: abast
'''

class Evaluator_Setup:
    def __init__(self):
        self.evaluate_funs = []
        self.finalize_funs = []

class Evaluator:
    def __init__(self):
        #self.objectives = objectives
        self.setup = Evaluator_Setup()
        
    def evaluate(self, features_dict):
        ret = {}
        for in_name, fun, out_name in self.setup.evaluate_funs:
            ret[out_name] = fun(**features_dict[in_name])
        for fun in self.setup.finalize_funs:
            ret = fun(ret)
        return ret