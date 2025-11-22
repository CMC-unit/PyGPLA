
from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "PyGPLA"
author = "PyGPLA Developers"
copyright = f"{datetime.now().year}, {author}"

try:
    release = version("pygpla")
except PackageNotFoundError:
    release = "0.0.1"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".md": "myst",
    ".rst": "restructuredtext",
}
master_doc = "index"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Optional: tweak napoleon parsing of Google/Numpy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
