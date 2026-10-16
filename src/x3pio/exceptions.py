# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Exceptions raised by x3pio."""

from __future__ import annotations

__all__ = ["X3pChecksumError", "X3pError", "X3pFormatError"]


class X3pError(Exception):
    """Base class for all x3pio errors."""


class X3pFormatError(X3pError, ValueError):
    """Raised when file content or a value given for a file is not valid in x3p.

    It is a :class:`ValueError`, like :class:`json.JSONDecodeError`, so code that catches that
    also catches a bad value passed to x3pio.
    """


class X3pChecksumError(X3pFormatError):
    """Raised when an MD5 checksum stored in an x3p file does not match its content."""
