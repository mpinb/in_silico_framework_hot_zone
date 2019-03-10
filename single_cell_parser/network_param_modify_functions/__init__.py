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