import bluepyopt as bpop
import numpy
import deap
from bluepyopt.deapext import algorithms
import Interface as I
def robust_int(x):
    try: 
        return int(x)
    except:
        return None
    
def save_result(mdb_run, features, objectives):
    keys = [robust_int(x) for x in mdb_run.keys() if robust_int(x) is not None]
    if not keys:
        current_key = 0
    else:
        current_key = max(keys) + 1
    mdb_run.setitem(str(current_key), 
                    I.pd.concat([objectives, features], axis = 1), 
                    dumper = I.dumper_pandas_to_msgpack)

def setup_mdb_run(mdb_setup, run):
    '''mdb_setup contains a sub mdb for each run of the full optimization. This sub mdb is created here'''
    if not str(run) in mdb_setup.keys():
        mdb_setup.create_sub_mdb(str(run))
    mdb_run = mdb_setup[str(run)]
    mdb_run['0'] = ''
    if not 'checkpoint' in mdb_run.keys():
        mdb_run.create_managed_folder('checkpoint')  
    return mdb_run
         
def get_objective_function(mdb_setup):
    parameter_df = mdb_setup['params']
    Simulator = mdb_setup['get_Simulator'](mdb_setup)
    Evaluator = mdb_setup['get_Evaluator'](mdb_setup)
    def objective_function(param_values):
        p = I.pd.Series(param_values, index = parameter_df.index)
        s = Simulator.run(p)
        e = Evaluator.evaluate(s) 
        # ret is not a list but a dict!
        # however, bluepyopt expects a list. The conversion is performed in a custom mymap function
        # in the mymap function, the result is also stored
        return e
    return objective_function
    
def get_mymap(mdb_setup, mdb_run, c):
    # CAVE! get_mymap is doing more, than just to return a map function.
    # - the map function ignores the first argument. 
    #   The first argument (i.e. the function to be mapped on the iterable) is ignored.
    #   Instead, that function is hardcoded. It is defined in mdb_setup['get_Evaluator']
    #   This was neccessary, as my_ibea_evaluator and the deap individuals were causing pickle errors
    # - mymap also saves the features. As the bluepyopt optimizer only sees the sumarized 
    #   features (maximum of several individual features, see Hay et. al.), mymap is the ideal 
    #   place to insert a routine that saves all features before they get sumarized.
    # - mymap also creates a sub mdb in mdb_setup. The name of the sub_mdb is specified by n: It is str(n)
    #   mdb_setup[str(n)] then contains all the saved results    objective_fun = get_objective_function(mdb_setup)
    combiner = mdb_setup['get_Combiner'](mdb_setup)
    params = mdb_setup['params'].index
    objective_fun = get_objective_function(mdb_setup)
    def mymap(func, iterable):
        params_list = map(list, iterable)
        params_pd = I.pd.DataFrame(params_list, columns = params)
        features_dicts = c.gather(c.map(objective_fun, params_list))
        features_pd = I.pd.DataFrame(features_dicts)
        save_result(mdb_run, params_pd, features_pd)
        combined_objectives_dict = map(combiner.combine, features_dicts)
        combined_objectives_lists = [[d[n] for n in combiner.setup.names] 
                                     for d in combined_objectives_dict]
        return numpy.array(combined_objectives_lists)
    return mymap

class my_ibea_evaluator(bpop.evaluators.Evaluator):

    """Graupner-Brunel Evaluator"""

    def __init__(self, parameter_df, n_objectives):
        """Constructor"""
        super(my_ibea_evaluator, self).__init__()
        assert(isinstance(parameter_df, I.pd.DataFrame)) # we rely on the fact that the dataframe has an order
        self.parameter_df = parameter_df
        self.params = [bpop.parameters.Parameter
                       (index, bounds=(x['min'], x['max']))
                       for index, x in parameter_df.iterrows()]

        self.param_names = list(parameter_df.index)
        self.objectives = [1]*n_objectives 

    def evaluate_with_lists(self, param_values):
        return None # because of serialization issues, evaluate with lists is unused. 
        # Instead, the mymap function defines, which function is used to evaluate the parameters!!!



############################################################
# the following is taken from bluepyopt.deapext.algorithms.py
# it is changed such that the checkpoint only saves the population, but not the history 
# and hall of fame, as this can be easily reconstructed from the data saved by 
# mymap
############################################################

"""Optimisation class"""

"""
Copyright (c) 2016, EPFL/Blue Brain Project
 This file is part of BluePyOpt <https://github.com/BlueBrain/BluePyOpt>
 This library is free software; you can redistribute it and/or modify it under
 the terms of the GNU Lesser General Public License version 3.0 as published
 by the Free Software Foundation.
 This library is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 details.
 You should have received a copy of the GNU Lesser General Public License
 along with this library; if not, write to the Free Software Foundation, Inc.,
 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

# pylint: disable=R0914, R0912


import random
import logging

import deap.algorithms
import deap.tools
import pickle

logger = logging.getLogger('__main__')


def _evaluate_invalid_fitness(toolbox, population):
    '''Evaluate the individuals with an invalid fitness
    Returns the count of individuals with invalid fitness
    '''
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    return len(invalid_ind)


def _update_history_and_hof(halloffame, history, population):
    '''Update the hall of fame with the generated individuals
    Note: History and Hall-of-Fame behave like dictionaries
    '''
    if halloffame is not None:
        halloffame.update(population)

    history.update(population)


def _record_stats(stats, logbook, gen, population, invalid_count):
    '''Update the statistics with the new population'''
    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=gen, nevals=invalid_count, **record)


def _get_offspring(parents, toolbox, cxpb, mutpb):
    '''return the offsprint, use toolbox.variate if possible'''
    if hasattr(toolbox, 'variate'):
        return toolbox.variate(parents, toolbox, cxpb, mutpb)
    return deap.algorithms.varAnd(parents, toolbox, cxpb, mutpb)


def eaAlphaMuPlusLambdaCheckpoint(
        population,
        toolbox,
        mu,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        cp_frequency=1,
        cp_filename=None,
        continue_cp=False):
    r"""This is the :math:`(~\alpha,\mu~,~\lambda)` evolutionary algorithm
    Args:
        population(list of deap Individuals)
        toolbox(deap Toolbox)
        mu(int): Total parent population size of EA
        cxpb(float): Crossover probability
        mutpb(float): Mutation probability
        ngen(int): Total number of generation to run
        stats(deap.tools.Statistics): generation of statistics
        halloffame(deap.tools.HallOfFame): hall of fame
        cp_frequency(int): generations between checkpoints
        cp_filename(ModelDataBase or None): mdb_run, where the checkpoint is stored in [generation]_checlpoint. Was: path to checkpoint filename
        continue_cp(bool): whether to continue
    """
    # added by arco
    if cp_filename is not None:
        assert isinstance(cp_filename, I.ModelDataBase)
    if continue_cp:
        raise NotImplementedError()
    assert(halloffame is None)
    # end added by arco

    if continue_cp:
        # A file name has been given, then load the data from the file
        cp = pickle.load(open(cp_filename, "r"))
        population = cp["population"]
        parents = cp["parents"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        history = cp["history"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        start_gen = 1
        parents = population[:]

        ## commented out by arco ... as we record every evaluation, we do not need the bluepyopt history as it slows down the iteration
        #logbook = deap.tools.Logbook()
        #logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        #history = deap.tools.History()
        ## end commented out by arco

        # TODO this first loop should be not be repeated !
        invalid_count = _evaluate_invalid_fitness(toolbox, population)
        ## commented out by arco ... as we record every evaluation, we do not need the bluepyopt history as it slows down the iteration
        #_update_history_and_hof(halloffame, history, population)
        #_record_stats(stats, logbook, start_gen, population, invalid_count)
        ## end commented out by arco


    # Begin the generational process
    for gen in range(start_gen + 1, ngen + 1):
        offspring = _get_offspring(parents, toolbox, cxpb, mutpb)

        population = parents + offspring

        invalid_count = _evaluate_invalid_fitness(toolbox, offspring)
        ## commented out by arco
        #_update_history_and_hof(halloffame, history, population)
        #_record_stats(stats, logbook, gen, population, invalid_count)
        ## end commented out by arco

        # Select the next generation parents
        parents = toolbox.select(population, mu)

        ## commented out by arco
        #logger.info(logbook.stream)
        ## end commented out by arco

        if(cp_filename and cp_frequency and
           gen % cp_frequency == 0):
            cp = dict(population=population,
                      generation=gen,
                      parents=parents,
                      halloffame=None, #halloffame, // arco
                      history=None, # history, // arco
                      logbook=None, # logbook, // arco
                      rndstate=random.getstate())
            # save checkpoint in mdb
            cp_filename.setitem('{}_checkpoint'.format(gen), cp, dumper = I.dumper_to_pickle)
            #pickle.dump(cp, open(cp_filename, "wb"))
            #logger.debug('Wrote checkpoint to %s', cp_filename)

    return population #, logbook, history

####################################################
# end taken from bluepyopt
#####################################################

def run(self,
            max_ngen=10,
            offspring_size=None,
            continue_cp=False,
            cp_filename=None,
            cp_frequency=1,
            pop = None):
    
    """Copy pasted from the BluePyOpt package, however extended, such that a start population can be defined.
    Note: the population needs to be in a special format. Use methods in biophysics_fitting.population 
    to create a population."""
    # Allow run function to override offspring_size
    # TODO probably in the future this should not be an object field anymore
    # keeping for backward compatibility
    if offspring_size is None:
        offspring_size = self.offspring_size

    # Generate the population objec
    if pop is None:
        pop = self.toolbox.population(n=offspring_size)
    else:
        print "initialized with population of size %s" % len(pop)
        assert (continue_cp == False)

    ## commented out by arco
    #stats = deap.tools.Statistics(key=lambda ind: ind.fitness.sum)
    #import numpy
    #stats.register("avg", numpy.mean)
    #stats.register("std", numpy.std)
    #stats.register("min", numpy.min)
    #stats.register("max", numpy.max)
    ## end commented out by arco

    pop = eaAlphaMuPlusLambdaCheckpoint(
        pop,
        self.toolbox,
        offspring_size,
        self.cxpb,
        self.mutpb,
        max_ngen,
        stats=None, # arco
        halloffame= None, # self.hof,
        cp_frequency=cp_frequency,
        continue_cp=continue_cp,
        cp_filename=cp_filename)

    return pop

def start_run(mdb_setup, n, pop = None, client = None, 
              offspring_size=1000, eta=10, mutpb=0.3, cxpb=0.7, max_ngen = 600):
    '''function to start an optimization run as specified in mdb_setup. The following attributes need
    to be defined in mdb_setup:
    
    - params ... this is a pandas.DataFrame with the parameternames as index and the columns min_ and max_
    - get_Simulator ... function, that returns a biophysics_fitting.simulator.Simulator object
    - get_Evaluator ... function, that returns a biophysics_fitting.evaluator.Evaluator object.
    - get_Combiner ... function, that returns a biophysics_fitting.combiner.Combiner object
    
    For an exemplary setup of a Simulaotr, Evaluator and Combiner object, see 
    biophysics_fitting.hay_complete_default_setup'''
    # CAVE! get_mymap is doing more, than just to return a map function.
    # - the map function ignores the first argument. 
    #   The first argument (i.e. the function to be mapped on the iterable) is ignored.
    #   Instead, that function is hardcoded. It is defined in mdb_setup['get_Evaluator']
    #   This was neccessary, as my_ibea_evaluator and the deap individuals were causing pickle errors
    # - mymap also saves the features. As the bluepyopt optimizer only sees the sumarized 
    #   features (maximum of several individual features, see Hay et. al.), mymap is the ideal 
    #   place to insert a routine that saves all features before they get sumarized.
    # - mymap also creates a sub mdb in mdb_setup. The name of the sub_mdb is specified by n: It is str(n)
    #   mdb_setup[str(n)] then contains all the saved results
    mymap = get_mymap(mdb_setup, n, client)
    
    len_objectives = len(mdb_setup['get_Combiner'](mdb_setup).setup.names)
    parameter_df = mdb_setup['params']
    evaluator_fun = mdb_setup['get_Evaluator'](mdb_setup)
    evaluator = my_ibea_evaluator(parameter_df, len_objectives)
    mdb_run = setup_mdb_run(mdb_setup, n)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size,                                                                                                                                            
                                              eta=eta, mutpb=mutpb, cxpb=cxpb, 
                                              map_function = get_mymap(mdb_setup, mdb_run, client), 
                                              seed=n)
    pop = run(opt, 
                           max_ngen=max_ngen, 
                           cp_filename = mdb_run, 
                           pop = pop)
    return pop
