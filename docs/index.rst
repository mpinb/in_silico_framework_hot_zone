.. In-Silico Framework (ISF) documentation master file, created by
   sphinx-quickstart on Wed Mar 22 13:27:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to In-Silico Framework (ISF)'s documentation!
=====================================================

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

   tutorials/00_intro_to_tutorials.ipynb


.. nbgallery::
   :caption: 1. Data Analysis
   :glob:

   tutorials/1. data analysis/*

.. nbgallery::
   :caption: 2. biophysics
   :glob:

   tutorials/2. biophysics/*


.. nbgallery::
   :caption: 3. synaptic simulations
   :glob:

   tutorials/3. synaptic simulations/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`