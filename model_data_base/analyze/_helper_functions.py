import pandas as pd
import numpy as np

def map_return_to_series(fun, *args, **kwargs):
    def inner(*args, **kwargs):
        dummy = fun(*args, **kwargs)
        return pd.Series(dict(A = dummy))
    return inner

def is_int(x):
    '''checks an object is int. surpringingly fast.'''
    try:
        int(x)
        return True
    except:
        return False

def time_list_from_pd(syn):
    '''returns the columns, that can be interpreted as integer:
    'name1' 'nam2' '3' '1' '2' --> '3' '1' '2' 
    The underlying assupmtion is, that the dataframe syn consists
    of columsn like 'name', 'color', 'bla', which describe the data
    and columns like '1', '2', '3', which are the actual data.'''
    relevant_columns = [_ for _ in syn if is_int(_)]
    out = []
    for col in relevant_columns:
        dummy = syn[col]
        dummy = dummy.dropna()
        out.append(dummy)
    return pd.concat(out).values

def pd_to_array(x):
    '''converts pd dataframe to array.
    not very efficient ... use for small dataframes only.'''
    if x.empty:
        return np.array([]) 
    #x = pdframe.copy()
    array = []
    for lv in range(max(x.index.values)+1):
        if lv in list(x.index.values):
            array.append(x.loc[lv])
        else:
            array.append([0]*len(x.iloc[0]))            
    return np.array(array)  