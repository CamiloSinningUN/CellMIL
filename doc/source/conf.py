# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from typing import Any

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath("../../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cellmil"
copyright = "2025, CamiloSinning"
author = "CamiloSinning"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_default_options: dict[str, Any] = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_mock_imports = ["torch_cluster", "radiomics", "SimpleITK", "cucim"]


# Autosummary settings
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"

html_extra_head = [
    """
    <script>
    (function() {
        var e = document.documentElement;
        e.setAttribute("data-color-mode", "light");
        e.classList.remove("dark");
        e.classList.add("light");
    })();
    </script>
    """,
]

# Theme options
html_theme_options: dict[str, Any] = {
    "color_mode": "light",
    "accent_color": "blue",
    "globaltoc_expand_depth": 1,
    "github_url": "https://github.com/CamiloSinningUN/CellMIL",
    "nav_links": [
        {
            "title": "Installation",
            "url": "installation",
        },
        {
            "title": "Quick Start",
            "url": "quickstart",
        },
        {
            "title": "Pipeline",
            "url": "pipeline",
            "children": [
                {
                    "title": "Patch Extraction",
                    "url": "pipeline/patch_extraction",
                    "summary": "Extract patches from whole slide images.",
                },
                {
                    "title": "Cell Segmentation",
                    "url": "pipeline/cell_segmentation",
                    "summary": "Segment cells in extracted patches.",
                },
                {
                    "title": "Graph Creation",
                    "url": "pipeline/graph_creation",
                    "summary": "Create spatial graphs from segmented cells.",
                },
                {
                    "title": "Feature Extraction",
                    "url": "pipeline/feature_extraction",
                    "summary": "Extract morphological and radiomics features.",
                },
                {
                    "title": "Dataset Creation",
                    "url": "pipeline/dataset_creation",
                    "summary": "Create datasets for MIL training.",
                },
                {
                    "title": "Feature Visualization",
                    "url": "pipeline/feature_visualization",
                    "summary": "Visualize and analyze extracted features with statistical descriptors.",
                },
            ],
        },
        {
            "title": "Explainability",
            "url": "explainability",
            "summary": "Attention heatmaps and SHAP feature importance for MIL models.",
        },
        {
            "title": "API Reference",
            "url": "api/index",
            "children": [
                {
                    "title": "Models",
                    "url": "api/_autosummary/cellmil.models",
                    "summary": "Models used in the package.",
                },
                {
                    "title": "CLI",
                    "url": "api/_autosummary/cellmil.cli",
                    "summary": "Command-line interface for interacting with the package.",
                },
                {
                    "title": "Interfaces",
                    "url": "api/_autosummary/cellmil.interfaces",
                    "summary": "Interfaces used for configuration.",
                },
                {
                    "title": "Features",
                    "url": "api/_autosummary/cellmil.features",
                    "summary": "Features extracted from at cell level.",
                },
                {
                    "title": "Data",
                    "url": "api/_autosummary/cellmil.data",
                    "summary": "Patch extraction and WSI handling.",
                },
            ],
        },
    ],
}


html_title = "Documentation"
html_short_title = "Documentation"
