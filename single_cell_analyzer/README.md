# Single Cell Analyzer

This module provides code to analyze the results of single-cell simulations and its network embeddings. They are often used by the [single cell parser](../single_cell_parser/) module.
<span style="color: yellow"> Warning: </span> it is not recommended to change these modules, as they are heavily tested. Refactor/adapt at your own risk.

# membrane_potential_analysis

- Post-synaptic time histograms (PSTHs)
- Spike detection
- Simple statistics of membrane voltage

# currents

- Find time and voltage of max depolarisation
- Compute currents in the soma

# Synapse analysis

synalalysis.py

Provides methods for analyzing the synaptic connections of the cell:
- Activation times of synapses
- intracellular distances
- distances to synapses