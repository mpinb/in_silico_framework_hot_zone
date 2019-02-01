'''
Created on Nov 08, 2018

@author: abast
'''

import single_cell_parser as scp
from .parameters import param_selector
import time        


class Simulator_Setup:
    def __init__(self):
        self.cell_param_generator = None
        self.cell_param_modify_funs = []
        self.cell_generator = None
        self.cell_modify_funs = []
        self.stim_setup_funs = []
        self.stim_run_funs = []
        self.stim_response_measure_funs = [] 
        self.params_modify_funs = []                
    
    def check(self):
        if self.cell_param_generator is None:
            raise ValueError('cell_param_generator must be set')
        if self.cell_generator is None:
            raise ValueError('cell_generator must be set')  
        self._check_first_element_of_name_is_the_same(self.stim_setup_funs, self.stim_run_funs)
        self._check_first_element_of_name_is_the_same(self.stim_setup_funs, self.stim_response_measure_funs)
        
        #if not len(self.stim_setup_funs) == len(self.stim_result_extraction_fun):
        #    raise ValueError('run_fun must be set') 
        
    def _check_not_none(self, var, varname, procedure_descirption):
        if var is None:
            raise ValueError('{} is None after execution of {}. '.format(varname, procedure_descirption) + 
                             'Please check, that the function returns a value!')
            
    def _check_first_element_of_name_is_the_same(self, list1, list2):
        # lists are stim_setup_funs, stim_run_funs, response_measure_fun
        
        # extract names
        names1 = [x[0] for x in list1]
        names2 = [x[0] for x in list2]
                 
        # prefix
        prefix1 = [x.split('.')[0] for x in names1]
        prefix2 = [x.split('.')[0] for x in names2]
        
        assert(tuple(sorted(prefix1)) == tuple(sorted(prefix2)))
        
    def get_stims(self):
        return [x[0].split('.')[0] for x in self.stim_run_funs]
    
    def get_stim_setup_fun_by_stim(self, stim):
        l = [x for x in self.stim_setup_funs if x[0].split('.')[0] == stim]
        assert(len(l) == 1)
        return l[0]
    
    def get_stim_run_fun_by_stim(self, stim):
        l = [x for x in self.stim_run_funs if x[0].split('.')[0] == stim]
        assert(len(l) == 1)
        return l[0]   
    
    def get_stim_response_measure_fun(self, stim):
        l = [x for x in self.stim_response_measure_funs if x[0].split('.')[0] == stim]
        return l[0]     
    
    def get_cell_params(self, params):
        for name, fun in self.params_modify_funs:
            params = fun(params) 
        cell_param = self.cell_param_generator()
        for name, fun in self.cell_param_modify_funs:
            print name
            #print len(params), len(param_selector(params, name))            
            cell_param = fun(cell_param, params = param_selector(params, name))
            self._check_not_none(cell_param, 'cell_param', name)
        return cell_param      
        
    def get(self, params):
        '''this is the main interface, as it initializes a cell corresponding to
        the configuration'''
        cell_params = self.get_cell_params(params)
        cell = self.cell_generator(cell_params)
        for name, fun in self.cell_modify_funs:
            print name
            #print len(param_selector(params, name))
            cell = fun(cell, params = param_selector(params, name))
            self._check_not_none(cell, 'cell', name)
        return cell, params


class Simulator:
    def __init__(self):
        self.setup = Simulator_Setup()

    def get_simulated_cell(self, params, stim): 
        '''returns cell and parameters used to set up cell'''
        t = time.time()
        # get cell object with biophysics            
        cell, params = self.setup.get(params) 
        # set up stimulus
        name, fun = self.setup.get_stim_setup_fun_by_stim(stim)
        print name, param_selector(params, name)
        fun(cell, params = param_selector(params, name))
        # run simulation
        name, fun = self.setup.get_stim_run_fun_by_stim(stim)
        print name,param_selector(params, name)
        cell = fun(cell, params = param_selector(params, name))
        print "simulating {} took {} seconds".format(stim, time.time()-t)
        return cell, params

    def run(self, params):
        '''returns recordings as it is setup in self.setup'''
        self.setup.check()
        out = {}            
        for stim in self.setup.get_stims(): # , fun in self.setup.stim_setup_funs:
            cell, params = self.get_simulated_cell(params, stim)
            # extract result
            name, fun = self.setup.get_stim_response_measure_fun(stim)
            print name, param_selector(params, name)
            result = fun(cell, params = param_selector(params, name))
            del cell            
            out.update({name: result})
        return out


def run_fun(cell,
            T = 34.0, Vinit = -75.0, dt = 0.025, 
            recordingSites = [], tStart = 0.0, tStop = 250.0, 
            vardt = True, silent = True):
    from sumatra.parameters import NTParameterSet
    sim = {'T': T,
           'Vinit': Vinit,
           'dt': dt,
           'recordingSites': recordingSites,
           'tStart': tStart,
           'tStop': tStop}    
    scp.init_neuron_run(NTParameterSet(sim), vardt = vardt)
    return cell