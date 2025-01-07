# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UWVFTrefftz'
copyright = '2025, Manuel Pena'
author = 'Manuel Pena'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))



extensions = [
    'sphinx.ext.autodoc',    # For parsing docstrings
    'sphinx.ext.napoleon',   # For Google and NumPy style docstrings
    'sphinx.ext.viewcode',   # Adds links to source code
    'myst_parser',
    'sphinx.ext.mathjax'
]

myst_enable_extensions = ["dollarmath"]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"