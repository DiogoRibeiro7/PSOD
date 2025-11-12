# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------

project = 'PSOD - Pseudo-Supervised Outlier Detection'
copyright = '2024, Diogo Ribeiro'
author = 'Diogo Ribeiro'

# TODO: Use dynamic versioning
# from psod import __version__
# release = __version__
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    # TODO: Add more extensions
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.todo',
    # 'sphinx_autodoc_typehints',
    # 'nbsphinx',  # For Jupyter notebooks
    # 'sphinx_copybutton',  # For copy button in code blocks
]

# TODO: Configure autodoc
# autodoc_member_order = 'bysource'
# autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# TODO: Configure theme options
# html_theme_options = {
#     'navigation_depth': 4,
#     'collapse_navigation': False,
#     'sticky_navigation': True,
# }

# TODO: Add logo and favicon
# html_logo = '_static/logo.png'
# html_favicon = '_static/favicon.ico'

# -- Extension configuration -------------------------------------------------

# TODO: Configure intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# TODO: Configure Napoleon for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
