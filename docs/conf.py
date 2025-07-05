# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../backpropcount'))

project = 'backpropcount'
copyright = '2025, J. Renner, M. A. Wright, C. C. S. Pedroso, B. E. Cohen, A. Saha, K. Bouchard, P. Ercius, P. Denes, A. Goldschmidt'
author = 'J. Renner, M. A. Wright, C. C. S. Pedroso, B. E. Cohen, A. Saha, K. Bouchard, P. Ercius, P. Denes, A. Goldschmidt'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = ['numpy', 'h5py', 'matplotlib', 'scipy', 'torch']
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
