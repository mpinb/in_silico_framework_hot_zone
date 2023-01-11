# Single Cell Parser

This module provides methods for interacting with NEURON or Amira files in Python. It contains methods for parsing cell data, such as morphology and membrane properties; methods for converting cellular data to other formats; applying synapses to a cell morphology given certain biological constraints (see [synapse_mapper](./synapse_mapper.py) and [synapse](./synapse.py)) etc...

While it is possible to launch simulations with this module, it is easier to have high-level access to simulations by using the modules [simrun2](../simrun2/) and/or [simrun3](../simrun3/).

This module makes heavy use of [single cell analyser](../single_cell_analyzer/).
<span style="color: yellow"> Warning: </span> it is not recommended to change these modules, as they are heavily tested. Refactor/adapt at your own risk.