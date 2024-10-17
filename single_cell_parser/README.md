# Single Cell Parser

This module has the functionality to run single trials of simulations using NEURON. 

It contains methods for parsing cell data, such as morphology and membrane properties; methods for converting cellular data to other formats; applying synapses to a cell morphology given certain biological constraints (see [synapse_mapper](./synapse_mapper.py) and [synapse](./synapse.py)) etc...

It provides methods for interacting with NEURON or Amira files in Python. 

While it is possible to launch simulations with this module, to run many simulations in a scalable way (using a high-perfomance cluster) use [simrun](../simrun/).

> __Warning__: it is not recommended to change these modules, as they are heavily tested. Refactor/adapt at your own risk.
