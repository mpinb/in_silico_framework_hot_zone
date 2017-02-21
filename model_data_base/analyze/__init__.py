from spike_detection import spike_detection
import spaciotemporal_binning

from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]



color_cellTypeColorMap = {'L1': 'cyan', 'L2': 'dodgerblue', 'L34': 'blue', 'L4py': 'palegreen',\
                    'L4sp': 'green', 'L4ss': 'lime', 'L5st': 'yellow', 'L5tt': 'orange',\
                    'L6cc': 'indigo', 'L6ccinv': 'violet', 'L6ct': 'magenta', 'VPM': 'black',\
                    'INH': 'grey', 'EXC': 'red', 'all': 'black', 'PSTH': 'blue'}

excitatory = ['L6cc', 'L2', 'VPM', 'L4py', 'L4ss', 'L4sp', 'L5st', 'L6ct', 'L34', 'L6ccinv', 'L5tt']
inhibitory = ['SymLocal1', 'SymLocal2', 'SymLocal3', 'SymLocal4', 'SymLocal5', 'SymLocal6', 'L45Sym', 'L1', 'L45Peak', 'L56Trans', 'L23Trans']




def split_synapse_activation(sa, selfcheck = True, excitatory = excitatory, inhibiotry = inhibitory):
    '''Splits synapse activation in EXC and INH component.
    
    Assumes, that if the cell type mentioned in the column synapse_type is
    in the list Interface.excitatory, that the synapse is excitatory, else inhibitory.
    
    selfcheck: Default: True. If True, it is checked that every celltype is either
    asigned excitatory or inhibitory
    '''
    if selfcheck:
        celltypes = sa.apply(lambda x: x.synapse_type.split('_')[0], axis = 1).drop_duplicates()
        for celltype in celltypes:
            assert(celltype in excitatory + inhibitory)
            
    sa['EI'] = sa.apply(lambda x: 'EXC' if x.synapse_type.split('_')[0] in excitatory else 'INH', axis = 1)
    return sa[sa.EI == 'EXC'], sa[sa.EI == 'INH']