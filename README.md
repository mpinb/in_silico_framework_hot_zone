# The In-Silico-Framework (ISF)

**Contents**

1. [Installation](#installation)
2. [Usage](#usage)

## Installation

Every student needs to be able to synchronize their repository with https://github.com/research-center-caesar/in_silico_framework. Detailed instructions on how to install the repo are given in the [installer directory](./Installer/).


## Usage

The [Interface module](./Interface.py) provides high-level access to all submodules in the in-silico-framework. It provides acces to:
- The [NEURON simulator](https://www.neuron.yale.edu/neuron/)
- Data management tools via the [Model DataBase (mdb) module](./model_data_base/)
- Visualisation methods via the [Visualize module](./visualize/)
- and much more

A walkthrough of the capabilities of ISF is presented in the ["Getting Started" notebook](./getting_started/getting_started.ipynb). Core functionalities are repeated below.

<details><summary>Model Database</summary>
<p>

### Model DataBase (mdb)

```python
import Interface as I
I.ModelDataBase # main class of model_data_base
I.mdb_init_simrun_general.init # default method to initialize a model data base with existing simulation results
I.mdb_init_simrun_general.optimize # converts the data to speed optimized compressed binary format
```

</p>
</details>

<details><summary>Simulating</summary>
<p>
	
### Simulating

Running a simulation requires 3 things to be defined
1. A neuron morphology (hoc-morphology)
2. A biophysical description of the neuron morphology, i.e. the ion-channel distribution (parameter file)
3. Some input (current injection, synaptic input ...)

Creating a neuron to simulate on is done by means of parsing a parameter file (`.param` file) with [Single Cell Parser (scp)](./single_cell_parser/). This parameter file is read in as a nested dictionary that contains the biophysical parameters and the filename of a morphology file (`.hoc` file).

Defining a cell can be done as such:
```python
import Interface as I
parameter_file = I.os.path.join("<filename>.param")
cell_parameters = I.scp.build_parameters(parameter_file) # this is the main method to load in parameterfiles
cell = I.scp.create_cell(cell_parameters.neuron)
```

Running a simulation on the previously defined cell can be done like so:
```python
import neuron
h = neuron.h  # NEURON's python API
# let's define a pipette at the some
iclamp = h.IClamp(0.5, sec=cell.soma)
iclamp.delay = 150 # give the cell 150 ms to reach steady state
iclamp.dur = 5 # duration: 5ms rectangular pulse
amplitudes = [0.619, 0.793, 1.507] # amplitudes in nA
for amp in amplitudes:
	iclamp.amp = amp  # set the amplitude
	I.scp.init_neuron_run(cell_parameters.sim)  # run the simulation
```
	
![](./etc/Figures/VoltageResponse.png)

</p>
</details>
