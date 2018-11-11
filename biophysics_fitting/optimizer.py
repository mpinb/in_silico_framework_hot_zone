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

    stats = deap.tools.Statistics(key=lambda ind: ind.fitness.sum)
    import numpy
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log, history = algorithms.eaAlphaMuPlusLambdaCheckpoint(
        pop,
        self.toolbox,
        offspring_size,
        self.cxpb,
        self.mutpb,
        max_ngen,
        stats=stats,
        halloffame=self.hof,
        cp_frequency=cp_frequency,
        continue_cp=continue_cp,
        cp_filename=cp_filename)

    return pop, self.hof, log, history

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
    p, hof, log, hst = run(opt, 
                           max_ngen=max_ngen, 
                           cp_filename = mdb_run['checkpoint'].join('checkpoint.pickle'), 
                           pop = pop)
    return p, hof, log, hst