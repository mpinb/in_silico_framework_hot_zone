from functools import partial
import os
import pandas as pd
import numpy as np
import cloudpickle
import shutil
from .utils import get_vector_norm
from model_data_base.utils import silence_stdout

class RW:
    def __init__(self, df_seeds = None, param_ranges = None, 
                 params_to_explore = None, evaluation_function = None, 
                 MAIN_DIRECTORY = None, min_step_size = 0, max_step_size = 0.02, 
                 checkpoint_every = 100):
        '''Class to perform RW exploration from a seedpoint.
        
        df_seeds: pandas dataframe which contains the individual seed points as rows and 
            the parameters as columns
            
        param_ranges: pandas dataframe, which contains the parameters as rows and has a 
            "min_" and "max_" column denoting range of values this parameter may take
            
        params_to_explore: list of parameters that should be explored. If None, all parameters are explored.
            
        evaluation_function: takes one argument (a new parameter vector), returns 
            inside, evaluation. 
                inside: boolean that indicates if the parameter vector is within experimental constraits
                    (i.e. results in acceptable physiology) or not. 
                evaluation: dictionary that will be saved alongside the parameters. For example, this should contain
                    ephys features.
                    
        MAIN_DIRECTORY: output directory in which results are stored.'''
        self.df_seeds = df_seeds
        self.param_ranges = param_ranges
        self.MAIN_DIRECTORY = MAIN_DIRECTORY
        self.evaluation_function = evaluation_function
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.checkpoint_every = checkpoint_every
        self.all_param_names = list(self.param_ranges.index)
        if params_to_explore is None:
            self.params_to_explore = self.all_param_names
        else:
            self.params_to_explore = params_to_explore
        
    def run_RW_parallel(self, selected_seed, n_particles):
        pass
    
    def _normalize_params(self,p):
        assert(isinstance(p,pd.Series))
        assert(len(p) == len(self.all_param_names))
        min_ = self.param_ranges['min']
        max_ = self.param_ranges['max']
        return (p-min_)/(max_-min_)
    
    def _unnormalize_params(self, p):
        assert(isinstance(p,pd.Series))
        assert(len(p) == len(self.all_param_names))
        min_ = self.param_ranges['min']
        max_ = self.param_ranges['max']
        return p*(max_-min_)+min_
        
    def run_RW(self, selected_seedpoint, particle_id, seed = None):
        # get the parameters of the seed point (there might be more info in df_seeds than just the parameters)
        seed_point_for_exploration_pd = self.df_seeds[self.params_to_explore].iloc[selected_seedpoint]
        print(len(seed_point_for_exploration_pd))
        # normalize seed point parameters
        seed_point_for_exploration_normalized_pd = self._normalize_params(seed_point_for_exploration_pd)
        seed_point_for_exploration_normalized_selected_np = seed_point_for_exploration_normalized_pd[self.params_to_explore].values
        
        # set seed
        assert(seed is not None)
        np.random.seed(seed)    
        
        # set up folder structure
        print('My random number generator seed is', seed)
        OPERATION_DIR = os.path.join(self.MAIN_DIRECTORY, '{}/{}'.format(selected_seedpoint, particle_id))
        if not os.path.exists(OPERATION_DIR):
            os.makedirs(OPERATION_DIR)
        print('I am particle', particle_id, 'and I write to', OPERATION_DIR)
        
        # check if we start from scratch or if we resume an exploration
        iterations = [int(f.split('.')[0]) for f in os.listdir(OPERATION_DIR) if f.endswith('.pickle')]
        iterations = sorted(iterations,reverse=True)
        if len(iterations) == 0:
            print('So far nothing simulated, start from seedpoint', selected_seedpoint)
            p = seed_point_for_exploration_pd # p is pandas and the full vector and unnormalized
            iteration = 0
            inside, initial_evaluation = self.evaluation_function(p)
            assert(inside)
            initial_evaluation['inside'] = inside
            out = [initial_evaluation]  # out is what will be saved
        else:
            # search for last model inside the space, starting from the previous saved iteration
            for iteration in iterations:
                df_path = os.path.join(OPERATION_DIR, '{}.pickle'.format(iteration))
                print('Found preexisting RW, continue from there. Iteration', iteration)
                print('Loading file', df_path) 
                df = pd.read_pickle(df_path)   
                df = df[df.inside]
                try:
                    p = df.iloc[-1][self.all_param_names] # p is pandas and the full vector and unnormalized
                    break
                except IndexError:
                    print("didn't find a model inside the space, try previous iteration")
            
            out = []
            # set the random number generator to the latest state
            assert(max(iterations) == iterations[0])
            rngn_path = os.path.join(OPERATION_DIR, '{}.pickle.rngn'.format(iterations[0]))
            with open(rngn_path, 'rb') as f:
                rngn = cloudpickle.load(f)
            print('set state of random number generator')
            np.random.set_state(rngn)
            
            # set current iteration to follow up on the latest saved iteration
            iteration = iterations[0] + 1
            out = [] # out is what is saved 
        
        p_normalized = self._normalize_params(p)
        p_normalized_selected_np = p_normalized[self.params_to_explore].values
        
        # exploration loop
        print('exploration loop')
        while True:
            print('New loop. Current iteration', iteration)
            if iteration % self.checkpoint_every == 0 and iteration > 0:
                print('Saving')
                df_path = os.path.join(OPERATION_DIR, '{}.pickle'.format(iteration))
                df = pd.DataFrame(out)
                df.to_pickle(df_path + '.saving')
                with open(df_path + '.rngn', 'wb') as f:
                    cloudpickle.dump(np.random.get_state(), f)  
                # deal with the case that exploration was interupted while saving the dataframe
                shutil.move(df_path + '.saving', df_path) 
                out = [] # reset output after saving
                
            # this inner loop suggests new movements until the suggested step is within bounds
            print('Get new position')    
            n_suggestion = 0        
            while True:
                n_suggestion += 1
                movement = np.random.randn(len(self.params_to_explore))
                movement = movement/get_vector_norm(movement)
                step_size = np.random.rand()*(self.max_step_size-self.min_step_size)+self.min_step_size
                movement = movement * step_size
                p_proposal = p_normalized_selected_np + movement    
                if p_proposal.max() <= 1 and p_proposal.min() >= 0:
                    break
            print('Position within boundaries found, step size is', step_size, 
                  'Tested ', n_suggestion, 'positions to find one inside the box.') 
            
            # homogenize parameter representation
            # note p_proposal is normalized and numpy
            # p is pandas and the full vector and unnormalized
            p = seed_point_for_exploration_normalized_pd.copy()
            p[self.params_to_explore] = p_proposal
            p = self._unnormalize_params(p)
            
            # evaluate new point
            inside, evaluation = self.evaluation_function(p)
            print('Inside the space?', inside)
            evaluation['n_suggestion'] = n_suggestion
            evaluation['inside'] = inside
            out.append(evaluation)
            if inside:
                print('Moving current position to proposed position')
                p_normalized_selected_np = p_proposal
                print('distance to initial seed point (normalized):', 
                      get_vector_norm(p_normalized_selected_np-seed_point_for_exploration_normalized_selected_np))
            iteration += 1

## minimal running example of RW class

# df_seeds = I.pd.DataFrame({'model1':{'param1':1, 'param2':1}, 'model2':{'param1':0,'param2':1}}).T
# df_seeds
# param_ranges = I.pd.DataFrame({'param1':{'min':0, 'max':10}, 'param2':{'min':-10,'max':10}}).T
# param_ranges
# def evaluation_function(p):
#     print(p)
#     import time
#     #time.sleep(0.5)
#     if all(p.values == [1,1]):
#         return True, p.to_dict()
#     else:
#         return False, p.to_dict()
# 
# rw = RW(param_ranges=param_ranges,
#         df_seeds=df_seeds,
#         evaluation_function=evaluation_function,
#         MAIN_DIRECTORY='/gpfs/soma_fs/scratch/abast/testRW')
# 
# del param_ranges, df_seeds, evaluation_function

#################################
# efficient evaluation: interupt as soon as any objective is missed, i.e. evaluate incrementally
#################################

def evaluation_function_incremental_helper(p,
                                           s = None,  
                                           cutoffs = {'bAP':3.2, 
                                                  'BAC': 3.2, 
                                                  'StepOne':4.5, 
                                                  'StepTwo': 4.5, 
                                                  'StepThree': 4.5},
                                           stim_order = ['bAP', 'BAC', 'StepOne', 'StepTwo', 'StepThree'], 
                                           verbose = True,
                                           evaluators_by_stimulus = None,
                                           additional_evaluation_functions = [],
                                           objectives_by_stimulus = None):
    '''
    global variables: 
    evaluators_by_stimulus
    objectives_dict
    '''
    p = p.copy()
    evaluation = {}
    evaluation.update(p)
    voltage_traces = {}
    for stim in stim_order:
        if verbose:
            print('evaluating stimulus', stim)
        with silence_stdout:
            voltage_traces_ = s.run(p, stims = [stim])
            voltage_traces.update(voltage_traces_)
            # this is currently specific to the hay simulator / evaluator, which gets confused if 
            # any voltage traces beyond what it expects are present
            # thus filter it out and have a 'clean' voltage_traces_for_evaluation
            voltage_traces_for_evaluation = {k:v for k,v in voltage_traces.items() if k.endswith('hay_measure')}
            e = evaluators_by_stimulus[stim]
            evaluation_ = e.evaluate(voltage_traces_for_evaluation)
            evaluation.update(evaluation_)
        error = max(pd.Series(evaluation_)[objectives_by_stimulus[stim]])
        if error > cutoffs[stim]:
            if verbose: 
                print('stimulus', stim, 'has an error of', error, '- skipping further evaluation')
            #for k in full_evaluation_keys:
            #    if not k in evaluation:
            #        evaluation[k] = float('nan')
            return False, evaluation
    if verbose:
        print('all stimuli successful!')
    for aef in additional_evaluation_functions:
        evaluation.update(aef(voltage_traces))
    return True, evaluation