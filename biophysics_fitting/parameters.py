'''
Created on Nov 08, 2018

@author: abast
'''

import pandas as pd
import numpy as numpy
def param_to_kwargs(fun):
    '''returns a function that can be called with params = [some_Series].
    The content of Series is added to kwargs'''
    def fun2(*args, **kwargs):
        params = kwargs['params']
        del kwargs['params']
        kwargs.update(params.to_dict())
        return fun(*args, **kwargs)
    return fun2

def param_selector(params, s):
    '''Allows to select from a Series with an Index like a.b.c, a.b.d.

    params is a series with an index that contains strings seperated by "."
    Therefore, params can reflect a hierarchy, e.g.
        a.a   1
        a.b   2
        c.x   1
        c.a.b 7
    
    This method allows to select from that Series using a string s. E.g.
    
    param_selector(params, 'a'):
        > a    1
        > b    2
        
    param_selector(params, 'c.a'):
        > b    7
    '''
    split_char = '.'
    ssplit = s.split(split_char)    
    selection = pd.Series({k[len(ssplit[0]) + len(split_char):]: v 
                             for k, v in params.iteritems() 
                             if k.split(split_char)[0] == ssplit[0]})
    if len(ssplit) == 1:
        return selection
    elif len(ssplit) > 1:
        return param_selector(selection, split_char.join(ssplit[1:]))
    else:
        raise ValueError()

def set_fixed_params(params, fixed_params = None):
    return params.append(fixed_params)




##########################################
# tests
# todo: move in test suite
# currently automatically run on import ... should not be a problem since the evaluation is fast
################################################

def test_param_selectors():
    params = pd.Series({'a.a': 1, 'a.b': 2, 'b.x': 3, 'c.x': 1, 'c.a.b': 7})
    assert(len(param_selector(params, 'a')) == 2)
    assert(param_selector(params, 'a')['a'] == 1)
    assert(param_selector(params, 'a')['b'] == 2)

    assert(len(param_selector(params, 'b')) == 1)
    assert(param_selector(params, 'b')['x'] == 3)
    
    assert(len(param_selector(params, 'c')) == 2) 
    assert(len(param_selector(params, 'c.a')) == 1) 
    assert(param_selector(params, 'c.a')['b'] == 7)
test_param_selectors()

def test_param_to_kwargs():
    params = pd.Series({'a': 1, 'b': 2})
    def fun(**kwargs):
        assert(len(kwargs.keys()) == 2)
        assert(kwargs['a'] == 1)
        assert(kwargs['b'] == 2)
    param_to_kwargs(fun)(params = params)
test_param_to_kwargs()    

