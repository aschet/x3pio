# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Sphinx configuration for the x3pio documentation."""

from __future__ import annotations

import x3pio

project = "x3pio"
copyright = "2026, Thomas Ascher"
author = "Thomas Ascher"
release = x3pio.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_member_order = "bysource"

autodoc_default_options = {"members": True, "imported-members": True}

# Every reference has to resolve, except the array type of NumPy, which its documentation lists
# under another name.
nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy._typing._array_like.ArrayLike"),
    ("py:class", "ArrayLike"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_heading_anchors = 3

html_theme = "furo"
