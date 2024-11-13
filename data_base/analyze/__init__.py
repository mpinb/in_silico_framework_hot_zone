from .spike_detection import spike_detection
from . import spatiotemporal_binning
import logging
logger = logging.getLogger("ISF").getChild(__name__)
try:
    from barrel_cortex import excitatory, inhibitory
except ImportError:
    logger.warning("Could not import excitatory/inhibitory celltypes from barrel_cortex. Is the module available?")

def split_synapse_activation(
    sa,
    selfcheck=True,
    excitatory=None,
    inhibitory=None):
    '''Splits synapse activation in EXC and INH component.
    
    Assumes, that if the cell type mentioned in the column synapse_type is
    in the list Interface.excitatory, that the synapse is excitatory, else inhibitory.
    
    selfcheck: Default: True. If True, it is checked that every celltype is either
    asigned excitatory or inhibitory
    '''
    if excitatory is None:
        import barrel_cortex
        excitatory = barrel_cortex.excitatory
    if inhibitory is None:
        import barrel_cortex
        inhibitory = barrel_cortex.inhibitory
    if selfcheck:
        celltypes = sa.apply(lambda x: x.synapse_type.split('_')[0],
                             axis=1).drop_duplicates()
        for celltype in celltypes:
            assert celltype in excitatory + inhibitory

    sa['EI'] = sa.apply(
        lambda x: 'EXC'
        if x.synapse_type.split('_')[0] in excitatory else 'INH',
        axis=1)
    return sa[sa.EI == 'EXC'], sa[sa.EI == 'INH']