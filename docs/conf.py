import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------

project = 'SMINT'
copyright = '2026, Caroline Piaulet-Ghorayeb'
author = 'Caroline Piaulet-Ghorayeb'
master_doc = 'index'

release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'nbsphinx',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_logo = '../media/smint_logo.png'


html_static_path = []
