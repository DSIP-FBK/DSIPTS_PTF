# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import importlib.util
import os
import sys
from datetime import datetime

# Add the project root to the path so Sphinx can find the modules
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "DSIPTS"
copyright = f"2023-{datetime.now().year}, Andrea Gobbi, FBK"
author = "Andrea Gobbi"
release = "1.1.3"
version = "1.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate documentation from docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.mathjax",  # Render math via MathJax
    "sphinx.ext.intersphinx",  # Link to other project's documentation
    "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.githubpages",  # Create .nojekyll file for GitHub Pages
]

# Try to enable Markdown support (preferred: myst_parser, fallback: sphinx_mdinclude)
if importlib.util.find_spec("myst_parser"):
    extensions.append("myst_parser")
    # Configure myst_parser
    myst_enable_extensions = [
        "colon_fence",
        "deflist",
        "dollarmath",
        "fieldlist",
        "html_admonition",
        "html_image",
        "replacements",
        "smartquotes",
        "strikethrough",
        "substitution",
        "tasklist",
    ]
elif importlib.util.find_spec("sphinx_mdinclude"):
    extensions.append("sphinx_mdinclude")
else:
    print(
        "WARNING: Neither myst_parser nor sphinx_mdinclude available. "
        "Markdown files may not render correctly."
    )

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "private-members": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_preserve_defaults = True

# Mock imports for modules that might cause issues during doc generation
autodoc_mock_imports = [
    "aim",
    "hydra",
    "omegaconf",
    "lightning_utilities",
    "torchmetrics",
    "numba",
    "einops",
    "plotly",
    "beautifulsoup4",
    "html5lib",
    "html_table_parser_python3",
    "starlette",
    "pydantic",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# MathJax configuration
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# Templates and static files
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Master document
master_doc = "index"
root_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme configuration - Use sphinx_rtd_theme (Read the Docs theme)
try:
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
except ImportError:
    print("WARNING: sphinx_rtd_theme not available. Falling back to sphinx_pdj_theme.")
    try:
        import sphinx_pdj_theme

        html_theme = "sphinx_pdj_theme"
        html_theme_path = [sphinx_pdj_theme.get_html_theme_path()]
    except ImportError:
        print("WARNING: No theme available. Using default Sphinx theme.")
        html_theme = "alabaster"

# Theme options for sphinx_rtd_theme
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Static files
html_static_path = ["_static"]

# HTML output options
html_title = f"{project} {release} documentation"
html_short_title = project
html_show_sourcelink = True
html_show_sphinx = True
html_show_copyright = True

# Custom sidebar templates
html_sidebars = {
    "**": [
        "globaltoc.html",
        "relations.html",
        "sourcelink.html",
        "searchbox.html",
    ]
}

# Additional HTML options
html_copy_source = True
html_show_sourcelink = True
html_sourcelink_suffix = ""

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
}

# -- Options for manual page output ------------------------------------------

man_pages = [("index", "dsipts", "DSIPTS Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        "index",
        "dsipts",
        "DSIPTS Documentation",
        author,
        "dsipts",
        "Python library for time series forecasting.",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

# Autoclass content
autoclass_content = "both"
