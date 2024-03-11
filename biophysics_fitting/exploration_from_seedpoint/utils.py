import numpy as np
import pandas as pd
from data_base.utils import silence_stdout


def get_vector_norm(v):
    return np.sqrt(sum(v**2))

def evaluation_function_incremental_helper(p,
                                           s = None,  
                                           e = None,
                                           cutoffs = None,
                                           stim_order = None, 
                                           verbose = True,
                                           additional_evaluation_functions = [],
                                           objectives_by_stimulus = None):
    '''
    Allows to evaluate if a model shows responses with errors below the cutoff, one
    stimulus at a time. 
    
    Returns: True if all stimuli pass. False if any stimulus has an error above its cutoff. 
    
    s: Simulator object
    e: Evaluator object
    stim_order: order in which stimuli are simulated. List consisting of strings 
        and tuples of strings. Use strings if only one stimulus is to be simulated, 
        use tuples of strings to simulate several stimuli in one go. 
    cutoffs: dictionary, with keys that are in stim_order. Values are float and 
        indicate the maximum error allowed for these stimuli
    objectives_by_stimulus: dictionary with keys that are in stim_order. Values are lists 
        of names of objectives, returned by the evaluator object.
    additional_evaluation_functions: additional functions to be applied onto the final voltage 
        traces dictionary, which return a dictionary which is appended to the
        evaluations. 
    '''
    # make sure all defined cutoffs can actually be applied
    for c in cutoffs:
        assert(c in stim_order)
        assert(c in objectives_by_stimulus)
    p = p.copy()
    evaluation = {}
    evaluation.update(p)
    voltage_traces = {}
    for stim in stim_order:
        if verbose:
            print('evaluating stimulus', stim)
        with silence_stdout:
            voltage_traces_ = s.run(p, stims = stim)
            voltage_traces.update(voltage_traces_)
            # this is currently specific to the hay simulator / evaluator, which gets confused if 
            # any voltage traces beyond what it expects are present
            # thus filter it out and have a 'clean' voltage_traces_for_evaluation
            voltage_traces_for_evaluation = {k:v for k,v in voltage_traces_.items() if k.endswith('hay_measure')}
            evaluation_ = e.evaluate(voltage_traces_for_evaluation, raise_ = False)
            evaluation.update(evaluation_)
        if stim in cutoffs:
            error = max(pd.Series(evaluation_)[objectives_by_stimulus[stim]])
            if error > cutoffs[stim]:
                if verbose: 
                    print('stimulus', stim, 'has an error of', error, '- skipping further evaluation')
                return False, evaluation
    if verbose:
        print('all stimuli successful!')
    for aef in additional_evaluation_functions:
        evaluation.update(aef(voltage_traces))
    return True, evaluation