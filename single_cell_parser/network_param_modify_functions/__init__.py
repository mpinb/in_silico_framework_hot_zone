#import Interface as I
import barrel_cortex
def change_ongoing_interval(n, factor = 1, pop = None):
    '''scales the ongoing frequency with a factor'''
    for c in n.network.keys():
        celltype, location = c.split('_')
        if not celltype in pop:
            continue
        x = n.network[c]
        if isinstance(x.celltype, str):
            assert(x.celltype == 'spiketrain')
            x.interval = x.interval * factor
        else:
            x.celltype.spiketrain.interval = x.celltype.spiketrain.interval * factor
            
def set_stim_onset(n, onset = None):
    '''changes the offset when pointcells get activated'''
    for c in n.network.keys():
        x = n.network[c]
        if isinstance(x.celltype, str):
            assert(x.celltype == 'spiketrain')
            continue
        else:
            x.celltype.pointcell.offset = onset
            
def change_glutamate_syn_weights(param, g_optimal = None, pop = barrel_cortex.excitatory):
    for key in param.network.keys():
        celltype = key.split('_')[0]
        if celltype in pop: # I.excitatory:
            index = [x for x in g_optimal.index if x in celltype]
            assert(len(index) == 1)
            g = g_optimal[index[0]]
            param.network[key].synapses.receptors.glutamate_syn.weight = [g,g]
            
def change_evoked_INH_scaling(param, factor, pop = barrel_cortex.inhibitory):
    for key in param.network.keys():
        if key.split('_')[0] in pop:
            if param.network[key].celltype == 'spiketrain':
                continue
            prob = param.network[key].celltype.pointcell.probabilities
            prob = map(lambda x: x * factor, prob)
            param.network[key].celltype.pointcell.probabilities = prob
            
def _celltype_matches(celltype_name, celltypes, columns):
    assert(isinstance(celltypes, list))
    assert(isinstance(columns, list))
    return  celltype_name.split('_')[0] in celltypes \
                and (celltype_name.split('_')[1] in columns or 'S1' in columns)

def _has_evoked(param, celltype):
    assert(celltype in param.network.keys())    
    x = param.network[celltype]
    try:
        x.celltype.pointcell.probabilities
        return True
    except:
        return False


def inactivate_evoked_activity_by_celltype_and_column(param, inact_celltypes, inact_column):
    for celltype in param.network.keys():
        if _celltype_matches(celltype, inact_celltypes, inact_column) and _has_evoked(param, celltype):
            x = param.network[celltype]
            x.celltype.pointcell.probabilities = [0]*len(x.celltype.pointcell.probabilities)

def inactivate_evoked_and_ongoing_activity_by_celltype_and_column(param, inact_celltypes, inact_column):
    for celltype in param.network.keys():
        if _celltype_matches(celltype, inact_celltypes, inact_column):
            del param['network'][celltype]

# testing
# todo: move to testing module
import getting_started

assert(has_evoked(np, 'L5tt_C2'))
assert(~has_evoked(np, 'L1_Beta'))
assert(_celltype_matches('L5tt_C2', ['L5tt'], ['S1']))
assert(_celltype_matches('L5tt_C2', ['L5tt', 'L4ss'], ['S1']))
assert(~_celltype_matches('L5tt_C2', ['L5tt', 'L4ss'], ['D2']))
assert(_celltype_matches('L5tt_C2', ['L5tt', 'L4ss'], ['D2', 'C2']))

param = I.scp.build_parameters(getting_started.networkParam)
inactivate_evoked_and_ongoing_activity_by_celltype_and_column(param, ['L5tt'], ['S1'])
assert('L5tt' not in {k.split('_')[0] for k in param.network.keys()})
param = I.scp.build_parameters(getting_started.networkParam)
inactivate_evoked_and_ongoing_activity_by_celltype_and_column(param, ['L4ss'], ['S1'])
assert('L5tt' in {k.split('_')[0] for k in param.network.keys()})
param = I.scp.build_parameters(getting_started.networkParam)
inactivate_evoked_and_ongoing_activity_by_celltype_and_column(param, ['L5tt'], ['C2'])
assert('L5tt_C2' not in param.network.keys())
assert('L5tt_D2' in param.network.keys())