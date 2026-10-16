# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The options of the constructors and of ``with_metadata``, and the checks of them."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from typing import Literal, TypedDict

from .._version import __version__
from .datatypes import (
    DataType,
)
from .geometry import CoordinateSystem, Placement
from .header import (
    Metadata,
    ProbingType,
    Revision,
)
from .storage import DataStorage

__all__ = ["ArrayOptions", "MetadataChanges", "PointOptions"]


class MetadataChanges(TypedDict, total=False):
    """The fields of the metadata (``Record2``) that :meth:`X3pFile.with_metadata` can change.

    Each field is optional; a field that is left out stays as it is. For
    the fields that may be absent, ``None`` removes the value.

    :param date: Date and time of creation.
    :param creator: Name of the person or institution that created the data.
    :param calibration_date: Date and time of the last calibration.
    :param comment: Free text describing the data set.
    :param manufacturer: Manufacturer of the instrument or software.
    :param model: Model of the instrument or software.
    :param serial: Serial number of the instrument; may be empty for software.
    :param version: Version number of the instrument, of the software, or of both.
    :param probing_type: Kind of probing system.
    :param probing_identification: Description of the probing system.
    """

    date: datetime | None
    creator: str | None
    calibration_date: datetime | None
    comment: str | None
    manufacturer: str
    model: str
    serial: str
    version: str
    probing_type: ProbingType | None
    probing_identification: str


_INSTRUMENT_FIELDS = ("manufacturer", "model", "serial", "version")


def _apply_metadata_changes(base: Metadata | None, changes: Mapping[str, object]) -> Metadata:
    """Return a copy of ``base`` (or of a blank record) with ``changes`` applied."""
    unknown = changes.keys() - MetadataChanges.__annotations__.keys()
    if unknown:
        raise TypeError(f"Unknown metadata field(s): {', '.join(sorted(unknown))}")
    metadata = deepcopy(base) if base is not None else Metadata(date=datetime.now(UTC))
    for name, value in changes.items():
        if name in _INSTRUMENT_FIELDS:
            setattr(metadata.instrument, name, value)
        elif name == "probing_type":
            metadata.probing_system.type = value  # type: ignore[assignment]
        elif name == "probing_identification":
            metadata.probing_system.identification = value  # type: ignore[assignment]
        else:
            setattr(metadata, name, value)
    return metadata


def _check_options(options: Mapping[str, object], allowed: Mapping[str, object]) -> None:
    """Refuse an option that the function does not have, which a ``**options`` would swallow."""
    unknown = options.keys() - allowed.keys()
    if unknown:
        raise TypeError(f"Unexpected option(s): {', '.join(sorted(unknown))}")


def _new_metadata(given: Metadata | None) -> Metadata:
    """Return the record of a new file: a copy of the one given, or x3pio as the software.

    The standard asks for the date of creation. A record without a date gets the current UTC
    time, since the file is made now.
    """
    if given is None:
        return Metadata.for_software("x3pio", __version__)
    metadata = deepcopy(given)
    if metadata.date is None:
        metadata.date = datetime.now(UTC)
    return metadata


class ArrayOptions(TypedDict, total=False):
    """The options of :meth:`Profile.from_array`, :meth:`Surface.from_array` and :func:`write`.

    :param z_scale: Increment of the stored heights, or ``"auto"`` (the default) for the
        smallest one that fits the heights into ``z_type`` without overflow (``1`` for a
        floating point type, which stores metres).
    :param z_type: Storage type of the heights, ``float64`` by default.
    :param metadata: ``Record2``; defaults to a record naming x3pio as software with the current
        UTC time. The metadata is copied, and gets the current UTC time as its date if it has none.
    :param storage: Whether to write the heights as XML text or binary, binary by default.
    :param placement: Where the view coordinate system lies in the global one, see
        :class:`Placement`. Without it the two systems are the same.
    :param coordinate_system: Required with ``placement``: the coordinate system that the heights
        are in. :attr:`CoordinateSystem.VIEW`: they are stored as they are, like ``z`` of a file
        that is read. :attr:`CoordinateSystem.GLOBAL`: they are global, so the z offset of the
        placement is taken off. That is impossible with a rotation, since the global heights of a
        rotated grid are not on a regular grid.
    :param revision: The revision of the standard to write, see :class:`Revision`; Amd 1:2020
        by default.
    """

    z_scale: float | Literal["auto"]
    z_type: DataType
    metadata: Metadata | None
    storage: DataStorage
    placement: Placement
    coordinate_system: CoordinateSystem
    revision: Revision


class PointOptions(TypedDict, total=False):
    """The options that :meth:`PointCloud.from_points` and :func:`write_points` share.

    :param data_type: Storage type of all three coordinates, ``float64`` by default.
    :param scale: Increment of the stored coordinates of each axis, or ``"auto"`` (the default)
        for the smallest one that fits the axis into ``data_type`` without overflow (``1`` for a
        floating point type).
    :param metadata: ``Record2``; defaults to a record naming x3pio as software with the current
        UTC time. The metadata is copied, and gets the current UTC time as its date if it has none.
    :param storage: Whether to write the coordinates as XML text or binary, binary by default.
    :param placement: Where the view coordinate system lies in the global one, see
        :class:`Placement`. Without it the two systems are the same.
    :param coordinate_system: Required with ``placement``: the coordinate system that the points
        are in. :attr:`CoordinateSystem.GLOBAL`: they are global, and the stored view coordinates
        are the inverse of the placement applied to them, so ``layer.points()`` returns them
        again. :attr:`CoordinateSystem.VIEW`: they are stored as they are, like ``x``, ``y`` and
        ``z`` of a file that is read.
    :param revision: The revision of the standard to write, see :class:`Revision`; Amd 1:2020
        by default.
    """

    data_type: DataType
    scale: float | Literal["auto"]
    metadata: Metadata | None
    storage: DataStorage
    placement: Placement
    coordinate_system: CoordinateSystem
    revision: Revision
