.. Overview of all file formats used in ISF, containing both proprietary and custom file formats.

File formats
============

.. sectnum::

.am
===

The Amira proprietary file format. See `here <https://www.csc.kth.se/~weinkauf/notes/amiramesh.html>`_ for more information.
This flexible format can be used to store 3D scalar meshes, 3D neuron morphology reconstructions, slice image data etc.

.hx
===
AMIRA proprietary file format for saving AMIRA projects.

.hoc
====
NEURON :cite:`hines2001neuron` file format for neuron morphologies. Documentation can be found `here <https://nrn.readthedocs.io/en/latest/guide/hoc_chapter_11_old_reference.html>`_.
Used for 3D morphology reconstructions. Can be read with :py:mod:`single_cell_parser.cell_parser`.

.con
====
ISF custom file format to store neuron connection data. To be used in conjunction with an associated `.syn` file and morphology.
It numbers each synapse, and links it to its associated presynaptic cell type and ID.

Example::

    # Anatomical connectivity realization file; only valid with synapse realization:
    # 86_L5_CDK20041214_nr3L5B_dend_PC_neuron_transform_registered_C2center_synapses_20150504-1611_10389.syn
    # Type - cell ID - synapse ID

    L6cc_A3 0       0
    L6cc_A3 1       1
    L6cc_A3 2       2
    L6cc_A3 3       3
    L6cc_A3 4       4
    L6cc_A3 4       5
    ...

.syn
====
ISF custom file format to store synapse data. To be used in conjunction with an associated `.con` file and morphology.
For each synapse, it provides the synapse type and location onto the post-synaptic cell.
The location is encoded as a section ID and x (a normalized distance along the section),
to be consistent with NEURON syntax.

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
The `.param` format is used to store activity data covering spike times and time bins, as seen in e.g. getting_started/example_data/functional_constraints/evoked_activity/

Example::
    ...

Network parameters
------------------


