# Biophysics fitting

Create biophysically detailed models. Makes heavy use of three classes:
- Simulator: generates voltage traces
- Evaluator: extract features from voltage traces
- Combiner: combines features

For a Simulator object s and a Evaluator object e and a Combiner object c, the typical usecase is:
```python
    voltage_traces_dict = s.run(params)
    features = e.evaluate(voltage_traces_dict)
    combined_features = c.combine(features)
```

## Simulator

This class can be used to transform a parameter vector into simulated voltage traces.

The usual application is to specify biophysical parameters in a parameter vector and simulate
current injection responses depending on these parameters.

For a simulator object s, the main functions are:
s.run(params): returns a dictionary with the specified voltagetraces for all stimuli
s.get_simulated_cell(params, stim): returns params and cell object for stimulus stim
s.setup.get(params):
    Returns a cell with set up biophysics
s.setup.get_cell_params(params): returns cell NTParameterSet structure used for the 
    single_cell_parser.create_cell. This is helpful for inspecting, what parameters 
    have effectively been used for the simulation
s.setup.get_cell_params_with_default_sim_prams(params, ...): returns complete neuron parameter file
    that can be used for further simulations, i.e. with the simrun module or with roberts scripts.

<details>
<summary>Program flow</summary>
<br>


The "program flow" can be split in two parts.
(1) creating a cell object with set up biophysics from a parameter vector
(2) applying a variety of stimuli to such cell objects

An examplary specification of such a program flow can be found in the module hay_complete_default_setup.

The pipeline for (1) is as follows:
params: provided by the user: I.pd.Series object (keys: parameter names, values: parameter values)  
&nbsp;&nbsp;&nbsp;&nbsp;--> apply param_modify_functions (takes and returns a parameter vector, can alter it in any way)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---- cell_param template  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cell_params_generator  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v&nbsp;&nbsp;&nbsp;&nbsp;v  
cell_params: nested parameter structure, created from modified parameters and a template  
&nbsp;&nbsp;&nbsp;&nbsp;--> apply cell_param_modify_functions (takes and returns a NTParameterSet object, can alter it in any way)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;cell_generator(cell_params)    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;v  
cell object: created from the modified cell_params object by calling the 'cell_generator'  
&nbsp;&nbsp;&nbsp;&nbsp;--> apply cell_modify_functions (takes and returns a cell object, can alter it in any way)

Caveat: Try to avoid the usage of cell_modify_functions. While this allows
for any possible modification, it can be difficult to reproduce the result
later, as the cell object is different from what is expected by the cell_param
object. If possible, try to fully specify the cell in the cell_param object. Here,
it is also possible to spcify cell modifying functions, see
single_cell_parser.cell_modify_functions.

What form do the functions need to have?
```python
def example_cell_param_template_generator():
    #return a I.scp.NTParameterSet object as template. Ideally, all parameters, that need to be
    #filled in from the pasrameter vector have the value None, because it is tested, that
    #all None values have been replaced. E.g.
    return I.scp.NTParameterSet({'filename': path_to_hoc_morphology, 'Soma': somatic biophysical parameters})
    
def cell_generator(cell_param):
    return I.scp.create_cell(cell_params)
    
def example_cell_param_modify_function(cell_param, params):
    """ do something to the cell param object depending on params """
    return cell_param
    
def example_cell_modify_function(cell, params):
    """ do something to the cell object depending on params """
    return cell
```
        
Such functions can be registered to the Simlator object. Each function is registered with a name.

Each function, that receive the parameter vector (i.e. cell_param_modify_funs and cell_modify_funs)
only see a subset of the parameter vector that is provided by the user. This subset is determined 
by the name of the function.

E.g. let's assume, we have the parameters:
```python
{'apical_scaling.scale': 2,
 'ephys.soma.gKv': 0.001,
 'ephys.soma.gNav': 0.01
 }
```

Then, a function that is registered under the name 'apical_scaling' would get the following parameters:
```python
{'scale': 2}
```

The function, that is registered under the name 'ephys' would get the following parameters:
```python
{'soma.gKv': 0.001,
 'soma.Nav': 0.01}
```
 
Usually, a simulation contains fixed parameters, e.g. the filename of the morphology. Such fixed
parameters can be defined 

How can pipeline (1) be set up?
```python
s = Simulator() # instantiate simulator object
s.setup # Simualtor_Setup object, that contains all elements defining the pipeline above
s.setup.cell_param_generator =  example_cell_param_template_generator
s.setup.cell_generator = cell_generator
s.setup.params_modify_funs.append('name_of_param_modify_fun', example_cell_param_modify_function)
s.setup.cell_param_modify_funs.append('name_of_cell_param_modify_fun', example_cell_param_modify_function)
s.setup.cell_modify_funs.append('name_of_cell_modify_fun', example_cell_modify_function)
```

The pipeline for (2) is as follows:
Let s be the Simulator object

params: provided by the user: I.pd.Series object
&nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;|
  s.setup.get(params): triggers pipeline (1), results in a biophysically set up cell
&nbsp;&nbsp;&nbsp;|
&nbsp;&nbsp;&nbsp;v
For each stimulus: 
- stim_setup_funs
- stim_run_funs
- stim_response_measure_funs
       
What form do the functions need to have?
```python
def stim_setup_funs(cell, params):
    """ set up some stimulus """
    return cell
    
def stim_run_fun(cell, params):
    """ run the simulation """
    return cell
    
def stim_response_measure_funs(cell, params):
    """ extract voltage traces from the cell """
    return result
```
    
How can pipeline (2) be set up?
The names for stim_setup_funs, stim_run_funs and stim_response_measure_funs need to start
with the name of the simulus followed by a dot. For each stimulus, each of the three
functions needs to be defined exatly once, e.g. you could do something like:

```python
s.setup.stim_setup_funs.append(BAC.stim_setup, examplary_stim_setup_function)
s.setup.stim_run_funs.append(BAC.run_fun, examplary_stim_run_function)
s.setup.stim_response_measure_funs.append(BAC.measure_fun, examplary_stim_response_measure_function)
```

A typical usecase is to use the fixed parameters to specify to soma distance for a 
voltage trace of the apical dendrite. E.g.
```python
{'BAC.measure_fun.recSite': 835,
'BAC.stim_setup.dist': 835}
```

You would need to make sure, that your examplary_stim_run_fun reads the parameter 'recSite'
and sets up the stimulus accordingly.

Often, it is easier to write functions, that do not accept a parameter vector, but instead
keyword arguments. E.g. it might be desirable to write the examplary_stim_setup_funs like this
def examplary_stim_setup_function(cell, recSite = None):
    # set up current injection at soma distance recSite
    return cell
    
Instead of:
```python
def examplary_stim_setup_function(cell, params):
    recSite = params['recSite']
    # set up current injection at soma distance recSite
    return cell
```
    
This can be done by using the params_wo_kwargs method in biophysics_fitting.parameters. You would register
the function as follows:
s.setup.stim_setup_funs.append(BAC.stim_setup, params_to_kwargs(examplary_stim_setup_function))

</details>

## Evaluator

This  class can be used to extract features from (usually) voltagetraces
of different stimuli. The voltage traces are (usually) computed with a Simulator 
object by calling its run method.
        
In an optimization the features returned by Evaluator.evaluate are saved together
with to corresponding parameter values.

Note: Combining features to reduce the number of objectives should be done with the Combiner object.

## Combiner

Internally, the Combiner iterates over all names of specified combinations. Each combination
is specified not only by a name of the combination, but also a list of names of the features,
that go into that combination. Each list of features is then combined by calling combinefun with that list.

