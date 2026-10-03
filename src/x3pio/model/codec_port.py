# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The port through which the file types reach the codec.

A file type reads and writes a container with ``loads``, ``dumps`` and the other methods that
belong to it. The model knows no zip archive and no XML, so it declares what it needs of a codec
here, and the package registers the implementation when it is imported, see ``x3pio.codec``. The
dependency thus points from the codec to the model, as everywhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

    from .files import X3pFile
    from .storage import DataStorage

__all__ = ["Codec", "codec", "register"]


class Codec(Protocol):
    """What the file types need of the code that stores their files."""

    def parse(self, data: bytes, *, verify: bool) -> dict[str, Any]:
        """Parse the content of an x3p file into the fields of a file type.

        The fields are the arguments of ``X3pFile.from_stacked``, with the feature type.
        """

    def records(self, file: X3pFile) -> np.ndarray:
        """Return the coded coordinates in the layout of a binary data file."""

    def serialize(self, file: X3pFile, storage: DataStorage | None) -> bytes:
        """Return the content of an x3p file of ``file``, in ``storage`` or its own form."""


_registered: Codec | None = None


def register(implementation: Codec) -> None:
    """Make ``implementation`` the codec of the file types, in place of a former one."""
    global _registered
    _registered = implementation


def codec() -> Codec:
    """Return the registered codec.

    :raises RuntimeError: If none is registered, which cannot happen after importing ``x3pio``.
    """
    if _registered is None:
        raise RuntimeError("No codec is registered; import x3pio before using its file types")
    return _registered
