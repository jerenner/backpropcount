.. backpropcount documentation master file

backpropcount - Back-Propagation Counting
=====================================================

This project provides tools for electron counting in pixelated silicon detectors, such
as in 4D-STEM, using back-propagation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Overview
--------
The package consists of two main modules:

- **counting**: Implements the back-propagation counting method with PyTorch optimization.
- **profile**: Extracts and fits Gaussian profiles to electron hits.

Installation
------------
Clone the repository and install dependencies:

.. code-block:: bash

   pip install numpy h5py matplotlib scipy torch

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`