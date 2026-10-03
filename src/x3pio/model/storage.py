# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Storage forms of the coordinates: binary file or XML text."""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["DataStorage"]


class DataStorage(Enum):
    """How an x3p file stores its coordinates.

    The standard recommends the binary form for data sets of more than
    10000 points, and allows it for smaller ones too.
    """

    XML = auto()
    BINARY = auto()
