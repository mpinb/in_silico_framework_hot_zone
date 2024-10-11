# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import ast, os
project_root = os.path.join(os.path.abspath(os.pardir))
sys.path.insert(0, project_root)
from docs.parse_notebooks import copy_and_parse_notebooks_to_docs
from functools import lru_cache

project = 'In-Silico Framework (ISF)'
copyright = '2023, Arco Bast, Amir Najafgholi, Maria Royo Cano, Rieke Fruengel, Matt Keaton, Bjorge Meulemeester, Omar Valerio'
author = 'Arco Bast, Amir Najafgholi, Maria Royo Cano, Rieke Fruengel, Matt Keaton, Bjorge Meulemeester, Omar Valerio'
release = '0.2.0-alpha'
version = '0.2.0-alpha'
## Make your modules available in sys.path

# copy over tutorials and convert links to python files to sphinx documentation directives
#   copy_and_parse_notebooks_to_docs(
#       source_dir=os.path.join(project_root, 'getting_started', 'tutorials'),
#       dest_dir=os.path.join(project_root, 'docs', 'tutorials')
#   )

from compatibility import init_data_base_compatibility
init_data_base_compatibility()  # make db importable before running autosummary or autodoc etc...


# -- General configuration ------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    # 'sphinx.ext.autosummary',  # Create neat summary tables
    'autoapi.extension',      # improvement over autodoc, but still requires autodoc
    'sphinx.ext.napoleon',     # Support for NumPy and Google style docstrings
    'sphinx_paramlinks',       # Parameter links
    'sphinx.ext.todo',         # To-do notes
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',  # Link to other project's documentation, for e.g. NEURON classes as attributes in docstrings
    'nbsphinx',                # For rendering tutorial notebooks
    'nbsphinx_link',           # For linking to sections in tutorial notebooks
    'sphinxcontrib.bibtex',    # For citations
    'sphinx.ext.mathjax',      # For math equations
]

autoapi_dirs = [project_root]
autoapi_type = "python"
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-module-summary",
]

rst_prolog = """
.. role:: summarylabel
"""

"""Configure modules, functions, methods, classes and attributes so that they are not documented by Sphinx."""

project_root = os.path.join(os.path.abspath(os.pardir))

def skip_member(app, what, name, obj, skip, options):
    """Skip members if they have the :skip-doc: tag in their docstring."""
    # Debug print to check what is being processed
    # print(f"Processing {what}: {name}")
    
    # skip special members, except __get__ and __set__
    if name.startswith('__') and name.endswith('__') and name not in ['__get__', '__set__']:
        return True
    
    # Skip if it has the :skip-doc: tag
    if obj.__doc__ and ':skip-doc:' in obj.__doc__:
        if ':skip-doc:' in obj.__doc__:
            # print(f"Docstring for {name}: {obj.__doc__}")
            print(f"Skipping {what}: {name} due to :skip-doc: tag")
            return True
    
    # Skip inherited members
    if hasattr(obj, '__objclass__') and obj.__objclass__ is not obj.__class__:
        return True
    
    modules_to_skip = find_modules_with_tag(project_root, tag=":skip-doc:")
    if name in modules_to_skip:
        print(f"Skipping {what}: {name} due to :skip-doc: tag in module {obj.__module__}")
        return True
    
    return skip
    
def get_module_docstring(module_path):
    """Get the docstring of a module without importing it."""
    try:
        # Find the module's file path
        print("Module path:", module_path)
        if not os.path.isfile(module_path):
            raise FileNotFoundError(f"Module file {module_path} not found")

        # Read the module's source code
        with open(module_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # Parse the source code
        parsed_ast = ast.parse(source_code)

        # Extract the docstring
        docstring = ast.get_docstring(parsed_ast)
        return docstring

    except Exception as e:
        print(f"Error getting docstring for module {module_path}: {e}")
        return None

@lru_cache(maxsize=None)
def find_modules_with_tag(source_dir, tag=":skip-doc:"):
    """Recursively find all modules with a specific tag in their docstring.
    
    Returns:
        List of module path glob patterns with the tag.
    """
    modules_with_tag = []

    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.endswith(".py"):
                module_path = os.path.join(root, f)
                docstring = get_module_docstring(module_path)
                if docstring and tag in docstring:
                    if "__init__" in module_path:
                        modules_with_tag.append(module_path.rstrip('__init__.py') + "**")
                    else:
                        modules_with_tag.append(module_path + "**")                

    return modules_with_tag

@lru_cache(maxsize=None)
def get_modules_to_skip():
    return ['**tests**', '**barrel_cortex**', '**installer**', '**__pycache__**'] + find_modules_with_tag(project_root, tag=":skip-doc:")

# Use the cached result
modules_to_skip = get_modules_to_skip()

# skipping documentation for certain members
print("ignoring modules: ", modules_to_skip)
autoapi_ignore = modules_to_skip


def contains(seq, item):
    """Jinja test to check if an item is a property (i.e. a class attribute that has __get__ and __set__)
    Used in the Jinja templates in _templates"""
    return item in seq

def setup(app):
    # skip members with :skip-doc: tag in their docstrings
    app.connect('autoapi-skip-member', skip_member)

def underline(s):
    line = "-"*len(s)
    return "{}\n{}".format(s, line)

def prepare_jinja_env(jinja_env) -> None:
    jinja_env.tests["contains"] = contains
    jinja_env.filters["underline"] = underline

autoapi_prepare_jinja_env = prepare_jinja_env

autoapi_generate_api_docs = True  # used in api_reference.rst to generate api stubs for the top-level modules.

bibtex_bibfiles = ['bibliography.bib']

# Napoleon settings
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False  # otherwise custom argument types will not work
napoleon_type_aliases = None
napoleon_attr_annotations = True

## Include Python objects as they appear in source files
## Default: alphabetically ('alphabetical')
# autodoc_member_order = 'bysource'

# autoclass_content = 'both'  # document both the class docstring, as well as __init__
## Generate autodoc stubs with summaries from code
# autosummary_generate = True
# autosummary_imported_members = False  # do not show all imported modules per module, this is too bloated
paramlinks_hyperlink_param = 'name'

# Don't run notebooks
nbsphinx_execute = 'never'
nbsphinx_codecell_lexer = "python"

# Add any paths that contain templates here, relative to this directory.
# We have custom templates that produce toctrees for modules and classes on module pages,
# and separate pages for classes
templates_path = ['_templates']
autoapi_template_dir = '_templates/autoapi'
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False  # less verbose for nested packages

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
sys.path.append(os.path.abspath("./_pygments"))
pygments_style = 'style.LightStyle'
pygments_dark_style = 'material'  # furo specific

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"  # sphinx_immaterial is nice, but requires more config

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "light_logo": "_images/isf-logo-black.png",
    "dark_logo": "_images/isf-logo-white.png",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#000000",  # black instead of blue
        "color-foreground-secondary": "#797979",  # slightly more muted than default
    },
    "dark_css_variables": {
        "color-brand-primary": "#fefaee",  # Off-white
        "color-brand-content": "#FFB000",  # Gold instead of dark blue
    },
}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = [
    'default.css',  # relative to html_static_path defined above
    'style.css',
    'downarr.svg',
    'css/custom.css'
]

html_js_files = [
    'overview.js'
]

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.

## I don't like links to page reST sources
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'
