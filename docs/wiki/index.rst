TabEval Documentation
=====================

Welcome to TabEval, a comprehensive Python framework for evaluating synthetic tabular data generation methods. 

Overview
--------

TabEval provides a unified interface for benchmarking various synthetic data generation algorithms across multiple 
evaluation dimensions including statistical fidelity, utility preservation, privacy protection, and structural consistency.

Key Features
~~~~~~~~~~~~

* **Comprehensive Evaluation Metrics**: 50+ evaluation metrics across multiple dimensions
* **Rich Plugin Ecosystem**: Support for 15+ state-of-the-art synthetic data generation methods
* **Flexible Benchmarking**: Easy-to-use benchmarking suite with caching and reproducibility features
* **Multi-Domain Support**: Tabular data, survival analysis, time series, images, and domain adaptation
* **Extensible Architecture**: Plugin-based design for easy integration of new methods and metrics

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install tabeval

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   installation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/metrics
   api/plugins
   api/benchmark
   api/utils

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
