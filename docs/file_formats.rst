============
File formats
============

.am
===

The Amira proprietary file format. See `here <https://www.csc.kth.se/~weinkauf/notes/amiramesh.html>`_ for more information.
This flexible format can be used to store 3D scalar meshes, 3D neuron morphology reconstructions, slice image data etc.

Readers:

- :py:mod:`~single_cell_parser.reader.read_scalar_field`
- :py:mod:`~single_cell_parser.reader.read_landmark_file`

.hx
===
AMIRA proprietary file format for saving AMIRA projects.

.. _hoc_file_format:

.hoc
====
NEURON :cite:`hines2001neuron` file format for neuron morphologies. Documentation can be found `here <https://nrn.readthedocs.io/en/latest/guide/hoc_chapter_11_old_reference.html>`_.
Used for 3D morphology reconstructions.

Readers:

- :py:mod:`~single_cell_parser.cell_parser`
- :py:meth:`~single_cell_parser.reader.read_hoc_file`

Example::

    {create soma}
    {access soma}
    {nseg = 1}
    {pt3dclear()}
    {pt3dadd(1.933390,221.367004,-450.045990,12.542000)}
    {pt3dadd(2.321820,221.046997,-449.989990,13.309400)}
    ...
    {pt3dadd(13.961900,210.149002,-447.901001,3.599700)}

    {create BasalDendrite_1_0}
    {connect BasalDendrite_1_0(0), soma(0.009696)}
    {access BasalDendrite_1_0}
    {nseg = 1}
    {pt3dclear()}
    {pt3dadd(6.369640, 224.735992, -452.399994, 2.040000)}
    {pt3dadd(6.341550, 222.962997, -451.906006, 2.040000)}
    ...

.. _mod_file_format:

.mod
====
NEURON :cite:`hines2001neuron` file format for neuron mechanisms. Documentation can be found `here <https://neuron.yale.edu/neuron/docs/using-nmodl-files>`_.
Used to define channel and synapse dynamics in NEURON simulations.
See the folder `mechanisms` in the project source.

.. _con_file_format:

.con
====
ISF custom file format to store connectivity data. 
To be used in conjunction with an associated :ref:`syn_file_format` file and morphology :ref:`hoc_file_format` file.
It numbers each synapse, and links it to its associated presynaptic cell type and ID.
While a :ref:`syn_file_format` file and :ref:`hoc_file_format` file provide the anatomical realization of a network,
the addition of a :ref:`con_file_format` file makes it into a functional realization, as it allows linking the synapses to
their presynaptic cells, which in turn allow cell type specific activation patters (see: :py:meth:`single_cell_parser.network.NetworkMapper`).

Readers:

- :py:mod:`~single_cell_parser.reader.read_functional_realization_map`

Example::

    # Anatomical connectivity realization file; only valid with synapse realization:
    # synapse_ralization_file.syn
    # Type - cell ID - synapse ID

    L6cc_A3 0       0
    L6cc_A3 1       1
    L6cc_A3 2       2
    L6cc_A3 3       3
    L6cc_A3 4       4
    L6cc_A3 4       5
    ...

.. _syn_file_format:

.syn
====
ISF custom file format to store synapse locations onto a morphology. 
This file fully captures an anatomical realization of a network.
Only valid with an associated morphology :ref:`_hoc_file_format` file.

For each synapse, it provides the synapse type and location onto the morphology.
Each row index corresponds to its synapse ID, providing a backlink to the :ref:`con_file_format` file format.
The location is encoded as a section ID and x (a normalized distance along the section),
to be consistent with NEURON syntax.

To create a functional network (i.e., known presynaptic origin), 
it must be used in conjunction with an associated :ref:`_con_file_format` file.

Readers:

- :py:mod:`~single_cell_parser.reader.read_synapse_realization`
- :py:mod:`~single_cell_parser.reader.read_pruned_synapse_realization`

Example::

    # Synapse distribution file
    # corresponding to cell: 86_L5_86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2center
    # Type - section - section.x

    VPM_E1  112     0.138046479525
    VPM_E1  130     0.305058053119
    VPM_E1  130     0.190509288017
    VPM_E1  9       0.368760777084
    VPM_E1  110     0.0
    VPM_E1  11      0.120662910562
    ...

.param
======
ISF custom file format to save JSON-like ASCII data. These can be read in using :py:mod:`single_cell_parser`.
Used in a variety of ways, as seen below.

Activity data
-------------
The `.param` format is used to store activity data covering spike times and time bins for specific cell types in response to a stimulus, as seen in e.g. getting_started/example_data/functional_constraints/evoked_activity/

Example::

    {
    "L4ss_B1": {
    "distribution": "PSTH",
    "intervals": [(0.0,1.0),(1.0,2.0),(2.0,3.0),(3.0,4.0),(4.0,5.0),(5.0,6.0),(6.0,7.0),(7.0,8.0),(8.0,9.0),(9.0,10.0),(10.0,11.0),(11.0,12.0),(12.0,13.0),(13.0,14.0),(14.0,15.0),(15.0,16.0),(16.0,17.0),(17.0,18.0),(18.0,19.0),(19.0,20.0),(20.0,21.0),(21.0,22.0),(22.0,23.0),(23.0,24.0),(24.0,25.0),(25.0,26.0),(26.0,27.0),(27.0,28.0),(28.0,29.0),(29.0,30.0),(30.0,31.0),(31.0,32.0),(32.0,33.0),(33.0,34.0),(34.0,35.0),(35.0,36.0),(36.0,37.0),(37.0,38.0),(38.0,39.0),(39.0,40.0),(40.0,41.0),(41.0,42.0),(42.0,43.0),(43.0,44.0),(44.0,45.0),(45.0,46.0),(46.0,47.0),(47.0,48.0),(48.0,49.0),(49.0,50.0)],
    "probabilities": [-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,0.0062,0.0062,-0.0004,0.0129,0.0062,-0.0004,-0.0004,0.0062,-0.0004,-0.0004,-0.0004,0.0062,0.0062,-0.0004,-0.0004,-0.0004],
    },
    "L4ss_B2": {
    "distribution": "PSTH",
    "intervals": [(0.0,1.0),(1.0,2.0),(2.0,3.0),(3.0,4.0),(4.0,5.0),(5.0,6.0),(6.0,7.0),(7.0,8.0),(8.0,9.0),(9.0,10.0),(10.0,11.0),(11.0,12.0),(12.0,13.0),(13.0,14.0),(14.0,15.0),(15.0,16.0),(16.0,17.0),(17.0,18.0),(18.0,19.0),(19.0,20.0),(20.0,21.0),(21.0,22.0),(22.0,23.0),(23.0,24.0),(24.0,25.0),(25.0,26.0),(26.0,27.0),(27.0,28.0),(28.0,29.0),(29.0,30.0),(30.0,31.0),(31.0,32.0),(32.0,33.0),(33.0,34.0),(34.0,35.0),(35.0,36.0),(36.0,37.0),(37.0,38.0),(38.0,39.0),(39.0,40.0),(40.0,41.0),(41.0,42.0),(42.0,43.0),(43.0,44.0),(44.0,45.0),(45.0,46.0),(46.0,47.0),(47.0,48.0),(48.0,49.0),(49.0,50.0)],
    "probabilities": [-0.0004,0.0062,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,0.0062,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004,0.0062,-0.0004,-0.0004,0.0129,0.0062,0.0062,-0.0004,-0.0004,-0.0004,-0.0004,0.0062,-0.0004,-0.0004,0.0062,-0.0004,-0.0004,-0.0004,-0.0004,-0.0004],
    },
    ...
    }

.. _network_parameters_format:

Network parameters
------------------
The `.param` format is used to store network parameters, containing information for each cell type in a network.
For each presynaptic cell type in the network, this following information is provided:

.. list-table:: Network Parameters
   :header-rows: 1

   * - Parameter
     - Description
   * - celltype
     - Spiking type of the presynaptic cell ("spiketrain", or "pointcell").
   * - interval
     - Average interval of the spikes.
   * - synapses
     - Additional synapse information (see below)
   * - releaseProb
     - Release probability of the synapse upon a spike.
   * - cellNr
     - Amount of connected presynaptic cells of this type.

The `synapse` key contains the following information:

- receptor type
- activation threshold
- activation delay
- rise and decay time dynamics (if applicable)
- weights
        
Example::

    {
    "info": {
        "date": "11Feb2015",
        "name": "evoked_activity",
        "author": "name",
    },
    "network": {
        "cell_type_1": {
            "celltype": "spiketrain",
            "interval": 2173.9,
            "synapses": {
                "receptors": {
                    "glutamate_syn": {
                        "threshold": 0.0,
                        "delay": 0.0,
                            "parameter": {
                            "tau1": 26.0,
                            "tau2": 2.0,
                            "tau3": 2.0,
                            "tau4": 0.1,
                            "decayampa": 1.0,
                            "decaynmda": 1.0,
                            "facilampa": 0.0,
                            "facilnmda": 0.0,
                            },
                        "weight": [1.47, 1.47],
                    },
                },
            "releaseProb": 0.6,
            "cellNr": 1,
            "noise": 0.0,
            "start": 0.0,
            "nspikes": 2,
            },
        },
        "cell_type_2": {
            "celltype": "pointcell",
            "distribution": "PSTH"
            "intervals": [(0, 10), (10, 20), (20, 40), (40, 50)],
            "probabilities": [0.0, 0.01, 0.05, 0.0],
            "offset": 0.0,
        },
        ...
    }


Dataframes
==========

.. _syn_activation_format:

Synapse activation
------------------

Writers:
    
    - :py:meth:`single_cell_parser.writer.write_synapse_activation_file`

Example:

    +---------------------+-------------+---------------+-------------+----------------+----------------+-------------------+
    | synapse type        | synapse ID  | soma distance | section ID  | section pt ID  | dendrite label | activation times  |
    +=====================+=============+===============+=============+================+================+===================+
    | presyn_cell_type_1  | 0           | 150.0         | 24          | 0              | 'basal'        | 10.2,80.5,140.8   |
    +---------------------+-------------+---------------+-------------+----------------+----------------+-------------------+
    | presyn_cell_type_1  | 1           | 200.0         | 112         | 0              | 'apical'       |                   |
    +---------------------+-------------+---------------+-------------+----------------+----------------+-------------------+
    | presyn_cell_type_2  | 2           | 250.0         | 72          | 0              | 'apical'       | 300.1,553.5       |
    +---------------------+-------------+---------------+-------------+----------------+----------------+-------------------+

.. _spike_times_format:

Spike times
-----------

