# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""File types, values and rules of x3p, independent of zip and XML.

This package depends on nothing but NumPy and the standard library. It knows no zip archive and
no XML, so everything in it can be used and tested without a file. The methods of a file type that
read and write a container (``loads``, ``dumps``, ``save``, ``raw_data``) import the codec
only when they are called; no module of the model imports it when it is loaded.
"""
