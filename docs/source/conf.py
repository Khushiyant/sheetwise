# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SheetWise'
copyright = '2025, Khushiyant Chauhan'
author = 'Khushiyant Chauhan'
release = '2.6.1' 

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.githubpages',
]

# Try to add optional extensions
try:
    import sphinx_autodoc_typehints
    extensions.append('sphinx_autodoc_typehints')
except ImportError:
    pass

try:
    import sphinx_copybutton
    extensions.append('sphinx_copybutton')
except ImportError:
    pass

try:
    import myst_parser
    extensions.append('myst_parser')
except ImportError:
    pass

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autosummary_generate = True
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
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = []

# Source file suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme options:
# 1. 'furo' - Modern, clean, best for Python projects (RECOMMENDED)
# 2. 'sphinx_rtd_theme' - Read the Docs (current)
# 3. 'pydata_sphinx_theme' - Used by NumPy, Pandas
# 4. 'sphinx_book_theme' - Book-like appearance
# 5. 'alabaster' - Sphinx default, minimal

html_theme = 'furo'
html_static_path = ['_static']

# Furo theme options (modern & clean)
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#ff6b6b",  # SheetWise accent color
        "color-brand-content": "#ff6b6b",
    },
    "dark_css_variables": {
        "color-brand-primary": "#ff8787",
        "color-brand-content": "#ff8787",
    },
}

# Alternative: Read the Docs theme options (uncomment to use)
# html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     'analytics_id': '',
#     'logo_only': False,
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': True,
#     'collapse_navigation': False,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }

# Alternative: PyData theme (uncomment to use)
# html_theme = 'pydata_sphinx_theme'
# html_theme_options = {
#     "icon_links": [
#         {
#             "name": "GitHub",
#             "url": "https://github.com/Khushiyant/sheetwise",
#             "icon": "fab fa-github-square",
#         },
#         {
#             "name": "PyPI",
#             "url": "https://pypi.org/project/sheetwise/",
#             "icon": "fas fa-box",
#         },
#     ],
#     "navbar_end": ["navbar-icon-links"],
# }

html_context = {
    "display_github": True,
    "github_user": "Khushiyant",
    "github_repo": "sheetwise",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

html_logo = None
html_favicon = None

# Additional settings
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'SheetWisedoc'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {}

latex_documents = [
    (master_doc, 'SheetWise.tex', 'SheetWise Documentation',
     'Khushiyant Chauhan', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'sheetwise', 'SheetWise Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'SheetWise', 'SheetWise Documentation',
     author, 'SheetWise', 'A Python package for encoding spreadsheets for Large Language Models.',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']
