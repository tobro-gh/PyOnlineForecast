# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyOnlineForecast'
copyright = '2026, tobro'
author = 'tobro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "numpydoc",
    "nbsphinx",
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",  # enables $...$ and $$...$$ math syntax
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
numpydoc_show_class_members = False

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',  # or 'alphabetical'
    'special-members': '__init__',
    'undoc-members': False,  # Don't show undocumented members
    'exclude-members': '__weakref__',
    'private-members': False,  # Hide _private methods
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# nbsphinx configuration
nbsphinx_execute = 'never'  # or 'auto' to execute during build
nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
