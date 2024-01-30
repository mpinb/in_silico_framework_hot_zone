import Interface as I
import pandas as pd
import os
import single_cell_parser as scp
import inspect
import six


def get_cellnumbers_from_confile(confile):
    con = I.scp.reader.read_functional_realization_map(confile)
    con = con[0]
    return {k: con[k][-1][1] + 1 for k in list(con.keys())}


def split_network_param_in_one_elem_dicts(dict_):
    out = []
    for k in list(dict_['network'].keys()):
        d = I.defaultdict_defaultdict()
        d['network'][k] = dict_['network'][k]
        out.append(I.scp.NTParameterSet(d))
    return out


def get_default_arguments(func):
    '''Gets the keyword arguments with their default value from any function.
    
    import six
    Returns: dictionary, function names are keys, default values are values'''
    o = inspect.getargspec(func)
    names = o.args[-len(six.get_function_defaults(func)):]
    defaults = o.defaults
    return {n: d for n, d in zip(names, defaults) if d is not None}


def set_default_arguments_if_not_set(o, kwargs):
    '''Used to update attributes of an object based on a dictionary.
    If the attribute is already set, it is NOT opverwritten.
    
    Usecase: If an object has been pickled and the keyword arguments have been extended post hoc, 
    the new keyword arguments are missing. This can be used to update the object accordingly.
    
    import six
    Returns: None'''
    for n, v in six.iteritems(kwargs):
        try:
            getattr(o, n)
        except AttributeError:
            errstr = 'Warning! Setting {} to default value {}'
            print(errstr.format(n, v))
            setattr(o, n, v)

def get_fraction_of_landmarkAscii(frac, path):
    'returns fraction of landmarkAscii files defined in path'
    f = os.path.basename(path)
    celltype = f.split('.')[-2]
    positions = scp.read_landmark_file(path)
    pdf = pd.DataFrame({'positions': positions, 'label': celltype})
    if len(pdf) == 0:  # cannot sample from empty pdf
        return pdf
    if frac >= 1:
        return pdf
    else:
        return pdf.sample(frac=frac)


def get_fraction_of_landmarkAscii_dir(frac, basedir=None):
    'loads all landmarkAscii files in directory and returns dataframe containing'\
    'position and filename (without suffix i.e. without .landmarkAscii)'
    out = []
    for f in os.listdir(basedir):
        if not f.endswith('landmarkAscii'):
            continue
        out.append(get_fraction_of_landmarkAscii(1, os.path.join(basedir, f)))

    return pd.concat(out).sample(frac=frac).sort_values('label').reset_index(
        drop=True)

def select_cells_that_spike_in_interval(
    sa,
    tmin,
    tmax,
    set_index=[
        'synapse_ID', 'synapse_type'
    ]):
    pdf = sa.set_index(list(set_index))
    pdf = pdf[[c for c in pdf.columns if c.isdigit()]]
    pdf = pdf[((pdf >= tmin) & (pdf < tmax)).any(axis=1)]
    cells_that_spike = pdf.index
    cells_that_spike = cells_that_spike.tolist()
    return cells_that_spike