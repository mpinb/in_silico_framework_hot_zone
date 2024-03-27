.. In-Silico Framework (ISF) documentation master file, created by
   sphinx-quickstart on Wed Mar 22 13:27:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The In-Silico Framework (ISF)
=====================================================

ISF is a multi-scale simulation environment for the generation, simulation, and analysis of neurobiologically tractable single cell and network-level simulations.

.. image:: ./_static/organigram@300x.png
  :width: 800px
  :alt: Overview of ISF

.. raw:: html
   <!DOCTYPE html>
   <head>
      <link rel="stylesheet" type="text/css" href="default.css">
   </head>


   <div class="row">
      <div class="header" id="in-vivo" width="10%">Input</div>
      <div class="fixed-cell" id="in-vivo" width="90%">In-vivo observation</div>
   </div>

   <div class="row" id="row2">
      <div class="header" id="in-vivo" width="10%">ISF</div>
      <div class="expandable-cell" id="neuron">
      <button type="button" class="collapsible" id="neuron-button">Neuron model</button>
      <div class="content" >
         <p>Lorem ipsum...</p>
      </div></div>

      <div class="expandable-cell" id="msm">
      <button type="button" class="collapsible" >Multi-scale model</button>
      <div class="content">
         <p>Lorem ipsum...</p>
      </div> </div>

      <div class="expandable-cell" id="network">
      <button type="button" class="collapsible" >Network model</button>
      <div class="content">
         <p>Lorem ipsum...</p>
      </div> </div>

   </div>

   <div class="row">
      <div class="header" id="in-vivo" width="10%">Output</div>
      <div class="fixed-cell">Mechanistic explanation</div>
   </div>


   <script src="./overview/overview.js"></script>



Module list
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   Interface
   barrel_cortex
   biophysics_fitting
   data_base
   simrun2
   simrun3
   single_cell_parser
   singlecell_input_mapper
   spike_analysis
   visualize

Tutorials
=============
.. nbgallery::
   :caption: Introduction
   :glob:

   Introduction_to_ISF.ipynb


.. nbgallery::
   :caption: 1. Neuron models
   :glob:

   tutorials/1. neuron models/*

.. nbgallery::
   :caption: 2. Network models
   :glob:

   tutorials/2. network models/*

.. nbgallery::
   :caption: 3. Multiscale models
   :glob:

   tutorials/2. network models/*


.. nbgallery::
   :caption: 4. Analytically tractable reduced models
   :glob:

   tutorials/4. reduced models/*

.. nbgallery::
   :caption: 5. Analysis
   :glob:

   tutorials/5. analysis/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`