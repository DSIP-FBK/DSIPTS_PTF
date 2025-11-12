.. DSIPTS documentation master file

DSIPTS Documentation
====================

**DSIPTS** (Deep Learning for Time Series) is a Python library for time series forecasting
that allows you to train state-of-the-art deep learning models on your time series data or
on benchmark datasets from the literature.

.. note::

   This project is under active development. Contributions and feedback are welcome!

Key Features
------------

* 🚀 **State-of-the-art Models**: Includes Transformer-based models (Informer, Autoformer, PatchTST, iTransformer),
  convolutional models, RNNs, and more
* 📊 **Flexible Data Handling**: Support for multi-variate time series with D1/D2 layer architecture
* ⚡ **PyTorch Lightning Integration**: Built on PyTorch Lightning for scalable training
* 🔧 **Easy Customization**: Add your own architectures and compare against existing models
* 📈 **Benchmark Datasets**: Built-in support for popular time series benchmarks

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/overview
   getting-started/installation
   getting-started/quickstart

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/data-pipeline
   user-guide/training
   user-guide/hydra-configs
   user-guide/installation
   user-guide/basic-usage
   user-guide/advanced-features

Reference
---------

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/d1-layer

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules
   dsipts.data_management
   dsipts.data_structure
   dsipts.models

Examples
--------

.. toctree::
   :maxdepth: 1
   :caption: Examples & Tutorials

   r_bash_examples
   r_dsipts

Development
-----------

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
