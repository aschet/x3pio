# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Reading and writing bytes from a path or a stream."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

__all__ = ["PathOrStream", "read_bytes", "write_bytes"]

PathLike = str | os.PathLike[str]
#: A path of a file, or an open binary stream.
PathOrStream = PathLike | IO[bytes]


def read_bytes(source: PathOrStream) -> bytes:
    """Return the content of a file given by its path, or read from an open binary stream."""
    if isinstance(source, (str, os.PathLike)):
        return Path(source).read_bytes()
    return source.read()


def write_bytes(destination: PathOrStream, data: bytes) -> None:
    """Write ``data`` to a file given by its path, or to an open binary stream."""
    if isinstance(destination, (str, os.PathLike)):
        Path(destination).write_bytes(data)
    else:
        destination.write(data)
