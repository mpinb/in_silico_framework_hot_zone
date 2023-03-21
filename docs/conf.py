import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.pardir)))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'In-Silico Framework (ISF)'
copyright = '2023, Arco Bast, Amir Najafgholi, Maria Royo Cano, Rieke Fruengel, Matt Keaton, Bjorge Meulemeester, Omar Valerio'
author = 'Arco Bast, Amir Najafgholi, Maria Royo Cano, Rieke Fruengel, Matt Keaton, Bjorge Meulemeester, Omar Valerio'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.napoleon'  # for NumPy and Google Doc style docstrings
    ]

# autosummary_generate=True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
# Sort members by type
autodoc_member_order = 'groupwise'
# Ensure that the __init__ method gets documented.
def skip(app, what, name, obj, skip, options):
    if name == "__init__":
        return False
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
