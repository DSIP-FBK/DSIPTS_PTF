# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import subprocess

# Installing required Sphinx extensions via pip
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sphinx_pdj_theme'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sphinx_mdinclude'])

import sphinx_pdj_theme
html_theme = 'sphinx_pdj_theme'
html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
sys.path.insert(0, os.path.abspath('..'))


project = 'dsipts'
copyright = '2023, Andrea Gobbi'
author = 'Andrea Gobbi'
release = '1.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Try to install sphinx_mdinclude if it's not already installed
try:
    import sphinx_mdinclude
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'sphinx_mdinclude'])

# Define extensions - conditionally include sphinx_mdinclude if available
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax'
]

# Try to add sphinx_mdinclude if available
try:
    import sphinx_mdinclude
    extensions.append('sphinx_mdinclude')
except ImportError:
    print("WARNING: sphinx_mdinclude not available. Some markdown files may not render correctly.")

mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autoclass_content = 'both'
source_suffix = ['.rst', '.md']