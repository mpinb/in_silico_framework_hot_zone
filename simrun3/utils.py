import Interface as I

def get_cellnumbers_from_confile(confile):
    con = I.scp.reader.read_functional_realization_map(confile)
    con = con[0]
    return {k: con[k][-1][1]+1 for k in con.keys()}

def split_network_param_in_one_elem_dicts(dict_):
    out = []
    for k in dict_['network'].keys():
        d = I.defaultdict_defaultdict()
        d['network'][k] = dict_['network'][k]
        out.append(I.scp.NTParameterSet(d))
    return out