# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The description of an x3p file that is read without its data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .extensions import VendorExtensions
from .geometry import Placement
from .header import FeatureType, Header, Metadata, Revision
from .storage import DataStorage

__all__ = ["FileDescription", "FileInfo"]


class FileDescription(Protocol):
    """The facts about an x3p file that do not depend on its coordinates.

    :class:`~x3pio.X3pFile` and :class:`FileInfo` both provide them, so a function that needs only
    these facts takes either. The members are read-only.
    """

    @property
    def header(self) -> Header:
        """The axes of ``Record1``."""
        ...

    @property
    def revision(self) -> Revision:
        """The revision of the standard that the file follows."""
        ...

    @property
    def placement(self) -> Placement:
        """Where the view coordinate system lies in the global one."""
        ...

    @property
    def feature_type(self) -> FeatureType:
        """The class of the data."""
        ...

    @property
    def storage(self) -> DataStorage:
        """Whether the coordinates are stored as XML text or binary."""
        ...

    @property
    def metadata(self) -> Metadata | None:
        """``Record2``, or ``None`` if the file has none."""
        ...

    @property
    def extensions(self) -> VendorExtensions:
        """The vendor specific extension files."""
        ...


@dataclass(frozen=True)
class FileInfo:
    """The description of an x3p file: everything except the coordinates and the validity bits.

    Returned by :func:`x3pio.info`. It provides a :class:`FileDescription` and adds the number and
    the shape of the layers. The fields cannot be assigned. ``metadata`` and ``extensions``
    are mutable objects, and are copies of what was read.

    :param header: The axes of ``Record1``.
    :param revision: The revision of the standard that the file follows.
    :param placement: Where the view coordinate system lies in the global one.
    :param feature_type: The class of the data.
    :param layer_count: The number of layers: one for a point cloud.
    :param shape: The shape of a layer: ``(rows, columns)`` of a surface, ``(points,)`` of a
        profile and of a point cloud.
    :param storage: Whether the coordinates are stored as XML text or binary.
    :param metadata: ``Record2``, or ``None`` if the file has none.
    :param extensions: The vendor specific extension files, as of a file that is read.
    """

    header: Header
    revision: Revision
    placement: Placement
    feature_type: FeatureType
    layer_count: int
    shape: tuple[int, ...]
    storage: DataStorage
    metadata: Metadata | None
    extensions: VendorExtensions
