# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""File types, values and rules of x3p, independent of zip and XML.

This package depends on nothing but NumPy and the standard library. It knows no zip archive and
no XML, so everything in it can be used and tested without a file. The methods of a file type that
read and write a container (``loads``, ``open``, ``dumps``, ``save``, ``raw_data``) call the codec
that ``x3pio`` registers through :mod:`x3pio.model.codec_port`; the model imports it nowhere.
"""
