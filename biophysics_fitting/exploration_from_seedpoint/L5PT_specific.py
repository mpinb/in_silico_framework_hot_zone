

### objectives ####
# init
import project_specific_ipynb_code.biophysical_models
objectives_BAC = project_specific_ipynb_code.biophysical_models.objectives_BAC
objectives_step = project_specific_ipynb_code.biophysical_models.objectives_step
objectives_2BAC = ['1BAC_APheight', '1BAC_ISI', '1BAC_ahpdepth', '1BAC_caSpike_height', '1BAC_caSpike_width', '1BAC_spikecount', '2BAC_APheight', '2BAC_ISI', '2BAC_ahpdepth', '2BAC_caSpike_height', '2BAC_caSpike_width', '2BAC_spikecount', 'bAP_APheight', 'bAP_APwidth', 'bAP_att2', 'bAP_att3', 'bAP_spikecount']

## below is added for incremental evaluation
objectives_bAP = ['bAP_APheight','bAP_APwidth','bAP_att2','bAP_att3','bAP_spikecount']
objectives_BAC2_only = ['1BAC_APheight', '1BAC_ISI', '1BAC_ahpdepth', '1BAC_caSpike_height', '1BAC_caSpike_width', '1BAC_spikecount', '2BAC_APheight', '2BAC_ISI', '2BAC_ahpdepth', '2BAC_caSpike_height', '2BAC_caSpike_width', '2BAC_spikecount']
objectives_step1 = [o for o in objectives_step if o.endswith('1')]
objectives_step2 = [o for o in objectives_step if o.endswith('2')]
objectives_step3 = [o for o in objectives_step if o.endswith('3')]

objectives_by_stimulus = {'bAP': objectives_bAP,
                   'BAC': objectives_BAC2_only, 
                   'StepOne': objectives_step1, 
                   'StepTwo': objectives_step2, 
                   'StepThree': objectives_step3}

### split up evaluator by objective ###

def get_evaluators_by_stimulus(evaluator):
    '''Takes an evaluator object, 
    returns a dictionary with one evaluator object per stimulus.
    
    Very L5PT specific in the way it deals with the BAC stimulus which can contain two BAC stimuli that are split
    and evaluated independently'''
    from copy import deepcopy
    stimuli = [x[0].split('.')[0] for x in evaluator.setup.evaluate_funs]
    evaluators_by_stimulus = {}
    for lv,stim in enumerate(stimuli):
        if 'BAC' in stim:
            continue
        e = deepcopy(evaluator)
        e.setup.evaluate_funs = [e.setup.evaluate_funs[lv]]
        # drop the function which splits the two consectuive bursts as this will only work
        # for the BAC stimulus
        e.setup.pre_funs = [e.setup.pre_funs[1]] 
        evaluators_by_stimulus[stim] = e
    e = deepcopy(evaluator)
    e.setup.evaluate_funs = [f for f in e.setup.evaluate_funs if 'BAC' in f[0]]
    evaluators_by_stimulus['BAC'] = e
    return evaluators_by_stimulus

### evaluator

