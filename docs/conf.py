"""Sphinx configuration for FAX."""

from fax import __version__

project = "FAX"
author = "UWPlasma contributors"
copyright = "2026, UWPlasma contributors"  # noqa: A001
version = release = __version__
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.mathjax", "sphinx.ext.napoleon"]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
html_static_path = ["_static"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
html_theme = "furo"
html_title = "FAX documentation"
html_theme_options = {
    "light_css_variables": {"color-brand-primary": "#0f5c8c", "color-brand-content": "#0f5c8c"},
    "dark_css_variables": {"color-brand-primary": "#6fb7e6", "color-brand-content": "#6fb7e6"},
}
