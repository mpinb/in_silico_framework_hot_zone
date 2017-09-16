import sys, os
from six import StringIO


def chunkIt(seq, num):
    '''splits seq in num lists, which have approximately equal size.
    https://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    '''
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    
    return [o for o in out if o] #filter out empty lists


def silence_stdout(fun, outfilename = None):
    '''robustly silences a function and restores stdout afterwars
    outfile: output can be redirected to file'''
    def silent_fun(*args, **kwargs):
        stdout_bak = sys.stdout
        if outfilename is None:
            sys.stdout = open(os.devnull, "w")
        else:
            sys.stdout = open(outfilename, 'w')
        try:
            res = fun(*args, **kwargs)
        except:
            raise
        finally:
            if outfilename:
                sys.stdout.close()
            sys.stdout = stdout_bak
        return res
    return silent_fun


def convertible_to_int(x):
        try:
            int(x)
            return True
        except:
            return False
        
def split_file_to_buffers(f, split_str = '#'):
    '''reads a file f and splits it, whenever "split_str" is found.
    Returns a list of StringIO Buffers.
    adapted from http://stackoverflow.com/a/33346758/5082048'''
    stringios = [] 
    stringio = None
    for line in f:
        if line.startswith(split_str):
            if stringio is not None:
                stringio.seek(0)
                stringios.append(stringio)
            stringio = StringIO()
        stringio.write(line)
        stringio.write("\n")
    stringio.seek(0)
    stringios.append(stringio)
    return stringios

def first_line_to_key(stringios):
    '''takes a list io StringIO objects. Each should contain one table.
    It returns a dictionary conatining the first line as key (assuming it is the name of the table)
    and the rest of it as value'''
    out = {}
    value = None
    for s in stringios:
        for lv, line in enumerate(s):
            if lv == 0:
                name = line.strip()
                value = StringIO()
            else:
                value.write(line)
                value.write("\n")
        value.seek(0)
        out[name] = value
    return out

from collections import defaultdict
import pandas as pd

def pandas_to_array(pdf, x_component_fun, y_component_fun, value_fun):
    '''this can convert a pandas dataframe, in which information
    is stored linearly to a 2D presentation.
    
    Example: you have a dataframe like:
               'bla'
    
    x_1_y_1    10
    x_2_y_1    15
    x_3_y_1    7
    x_1_y_2    2
    x_2_y_2    0
    x_3_y_2   -1
    
    Ans it should be converted to:
           1    2    3
    
    1      10   15   7
    2      2    0    -1
    3
    
    You can use:
    pandas_to_array(pdf, lambda index, values: index.split('_')[1], \
                         lambda index, values: index.split('_')[-1], \
                         lambda index, values: values.bla)
    '''
    out_dict = defaultdict(lambda: {})
    for index, values in pdf.iterrows():
        x = x_component_fun(index, values)
        y = y_component_fun(index, values)
        dummy = out_dict[x]
        assert(y not in dummy)
        dummy[y] = value_fun(index, values)
    
    return pd.DataFrame.from_dict(out_dict)


def select(df, **kwargs):
    for kwarg in kwargs:
        df = df[df[kwarg] == kwargs[kwarg]]
    return df

import numpy as np
def pooled_std(m, s, n):
    '''calculates the pooled standard deviation out of samples.
    
    m: means
    s: unbiased standarddeviation (normalized by N-1)
    n: number of samples per group
    
    returns: pooled mean, pooled std
    '''
    assert(len(m) == len(s) == len(n) > 0)
    M = np.dot(m,n) / float(sum(n))#[mm*nn / float(sum(n)) for mm, nn in zip(m,n)]
    # take carre of n = 0
    dummy = [(ss,mm,nn) for ss,mm,nn in zip(s,m,n) if nn >= 1]
    s,m,n = zip(*dummy) 
    assert(len(m) == len(s) == len(n) > 0)
    #calculate SD
    s = [ss * np.sqrt((nn-1)/float(nn)) for ss,nn in zip(s,n)] # convert to biased estimator  
    var_tmp = np.dot(n, [ss**2 + mm**2 for ss, mm in zip(s,m)]) / np.array(n).sum() - (np.dot(m, n) / float(sum(n)))**2 # calculate variance
    SD = np.sqrt(var_tmp) * np.sqrt(sum(n) /float(sum(n)-1)) #convert to unbiased estimator 
    return M,SD

def skit(*funcs, **kwargs):
    '''splits kwargs up to supply different functions with the right subset
    adapted from http://stackoverflow.com/a/23430335/5082048
    '''
    out = []
    for fun in funcs:
        out.append({key: value for key, value in kwargs.iteritems() 
                if key in fun.func_code.co_varnames})
        if 'kwargs' in fun.func_code.co_varnames:
            out[-1].update(kwargs)
        
    return tuple(out)

def unique(list_):
    return list(pd.Series(list_).drop_duplicates())

def cache(function):
    import cPickle, hashlib
    memo = {}
    def get_key(*args, **kwargs):
        return hashlib.md5(cPickle.dumps([args, kwargs])).hexdigest()
    def wrapper(*args, **kwargs):
        key = get_key(*args, **kwargs)
        if key in memo:
            return memo[key]
        else:
            rv = function(*args, **kwargs)
            memo[key] = rv
            return rv
    return wrapper
