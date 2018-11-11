'''
Created on Nov 08, 2018

@author: abast
'''

class Combiner_Setup:
    def __init__(self):
        self.combinations = []
        self.names = []
        self.combinefun = None
        
    def append(self, name, combination):
        self.combinations.append(combination)
        self.names.append(name)
        
class Combiner:
    def __init__(self):
        self.setup = Combiner_Setup()
        
    def combine(self, features):
        out = {}
        for n, c in zip(self.setup.names, self.setup.combinations):
            l = [features[k] for k in c]
            out[n] = self.setup.combinefun(l)
        return out