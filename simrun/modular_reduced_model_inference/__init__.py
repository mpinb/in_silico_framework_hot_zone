'''
This module implements a framework to 

What should I do if
 - I run my reduced model but the strategies are not executed remotely?
     --> make sure, the strategy has a Solver registered
     
     
Weba and pillow 2016?

Challenging of using not only temporal, but also spatial information -> did not work with just LDA.
Representation of input data is much larger.

Just time is numpy array of lenght 255 (stim) + 50 (stim) x n_trials (729k)

Including spatial extends with about x 50 to account for spatial bins.

==> data extractors.
'''

from .data_extractor import (
    DataExtractor_categorizedTemporalSynapseActivation,
    DataExtractor_spatiotemporalSynapseActivation,
    DataExtractor_daskDataframeColumn,
    DataExtractor_ISI,
    DataExtractor_object,
    DataExtractor_spikeInInterval,
    DataExtractor_spiketimes
)

from .solver import (
    Solver,
    Solver_COBYLA
)

from .reduced_model import (
    RaisedCosineBasis,
    Rm,
    DataView,
    DataSplitEvaluation,
    get_n_workers_per_ip    
)

from .strategy import (
    Strategy_categorizedTemporalRaisedCosine,
    Strategy_spatiotemporalRaisedCosine,
    Strategy_ISIcutoff,
    Strategy_ISIexponential,
    Strategy_ISIraisedCosine,
    Strategy_linearCombinationOfData,
    Strategy_temporalRaisedCosine_spatial_cutoff,
    CombineStrategies_sum,
    clear_memory,
    make_weakref,
    dereference,
    convert_to_numpy,
    _WEAKREF_ARRAY_LIST,
)