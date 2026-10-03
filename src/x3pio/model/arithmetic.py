# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Coordinate arithmetic on arrays: axis values, view points and global points.

The coordinates are in the view system: the stored values times the increment of the axis.
The :class:`~x3pio.Placement` of the file turns them into the global system. These functions
are the arithmetic behind the methods of the file types; see :mod:`x3pio.model.geometry` for the
concepts.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..exceptions import X3pFormatError
from .datatypes import DataType, get_data_type, suggest_scale
from .geometry import CoordinateSystem, Placement
from .header import Axis, AxisType, Header

__all__ = [
    "axis_values",
    "centered",
    "globalize",
    "rescaled",
    "resolve_coordinate_system",
    "view_points",
]


def axis_values(
    axis: Axis, stored: np.ndarray | None, shape: tuple[int, ...], index_axis: int
) -> np.ndarray:
    """Return the view coordinates of one axis for every point, shaped like the data.

    :param axis: The axis.
    :param stored: The coordinates of an absolute axis, ``None`` for an incremental one.
    :param shape: The shape of the data: a matrix of layers, rows and columns, or a list.
    :param index_axis: The axis of the data that an incremental axis counts along, ``-1``
        for x and ``-2`` for y.
    :raises X3pFormatError: If an absolute axis has no coordinates.
    """
    if not axis.is_incremental:
        if stored is None:
            raise X3pFormatError("An absolute axis needs its coordinates")
        return stored
    if len(shape) == 1:
        # A list has no matrix index: the coordinate is the index in the list.
        return np.arange(shape[0]) * axis.increment
    index_shape = [1] * len(shape)
    index_shape[index_axis] = shape[index_axis]
    index = (np.arange(shape[index_axis]) * axis.increment).reshape(index_shape)
    return np.broadcast_to(index, shape)


def view_points(
    header: Header, z: np.ndarray, x: np.ndarray | None, y: np.ndarray | None
) -> np.ndarray:
    """Return all points as an ``(N, 3)`` array in the view system, in storage order."""
    return np.stack(
        [axis_values(header.x, x, z.shape, -1), axis_values(header.y, y, z.shape, -2), z],
        axis=-1,
    ).reshape(-1, 3)


def resolve_coordinate_system(
    placement: Placement | None, coordinate_system: CoordinateSystem | None
) -> tuple[Placement, bool]:
    """Resolve the placement and the coordinate system that a new file is made with.

    :returns: The placement of the file, and whether the coordinates that were given are global,
        so that they have to be turned into view coordinates.
    :raises TypeError: If a placement is given without saying what the coordinates are in.
    """
    if placement is None:
        return Placement.identity(), False  # the two systems coincide
    if coordinate_system is None:
        raise TypeError(
            "A placement is given: say with coordinate_system whether the coordinates are global "
            "(CoordinateSystem.GLOBAL) or as stored in the file, in the view system "
            "(CoordinateSystem.VIEW)"
        )
    return placement, coordinate_system is CoordinateSystem.GLOBAL


def globalize(
    header: Header,
    placement: Placement,
    z: np.ndarray,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[Header, Placement, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Apply the placement to the coordinates.

    The arguments are not changed, and the result shares nothing with them. Without a rotation,
    the matrix and its regular grid are kept: the z offset is added to the
    heights, the offset of an incremental axis remains in the placement as the position of its
    first column or row, and an integer axis with an offset becomes ``float64``, since the
    shifted values no longer fall on multiples of its increment. With a rotation, the grid is
    skew, and the result has absolute axes with the coordinates of all points.

    :returns: The new header and placement, and the new ``z``, ``x`` and ``y``.
    """
    if placement.has_rotation:
        points = placement.apply(view_points(header, z, x, y)).reshape((*z.shape, 3))
        absolute = Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0)
        return (
            replace(header, x=absolute, y=replace(absolute), z=replace(absolute)),
            Placement.identity(),
            np.ascontiguousarray(points[..., 2]),
            np.ascontiguousarray(points[..., 0]),
            np.ascontiguousarray(points[..., 1]),
        )

    offset = placement.offset.copy()

    def baked(axis: Axis, index: int) -> Axis:
        if axis.is_incremental or offset[index] == 0.0:
            return replace(axis)  # an incremental axis keeps its offset in the placement
        offset[index] = 0.0
        if get_data_type(axis.data_type).is_integer:
            return replace(axis, data_type=DataType.FLOAT64, increment=1.0)
        return replace(axis)

    shifts = placement.offset
    return (
        replace(header, x=baked(header.x, 0), y=baked(header.y, 1), z=baked(header.z, 2)),
        Placement(offset=offset),
        z + shifts[2],
        None if x is None else x + shifts[0],
        None if y is None else y + shifts[1],
    )


def centered(
    header: Header,
    placement: Placement,
    z: np.ndarray,
    x: np.ndarray | None,
    y: np.ndarray | None,
) -> tuple[Placement, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Centre the stored coordinates of the absolute axes on the origin, keeping the global ones.

    Each absolute axis is shifted by the middle of its range, and the placement moves by the
    same shift, turned like the data: the global coordinates do not change. An incremental axis
    counts from the first point and is left alone.

    :returns: The new placement and the new ``z``, ``x`` and ``y``.
    """
    shift = np.array(
        [
            0.0 if x is None or header.x.is_incremental else _middle(x),
            0.0 if y is None or header.y.is_incremental else _middle(y),
            _middle(z),
        ]
    )
    moved = Placement._unchecked(placement.rotation, placement.rotation @ shift + placement.offset)
    return (
        moved,
        z - shift[2],
        None if x is None else x - shift[0],
        None if y is None else y - shift[1],
    )


def rescaled(header: Header, z: np.ndarray, x: np.ndarray | None, y: np.ndarray | None) -> Header:
    """Choose the increments of the integer axes anew, for the arrays that they are stored for.

    The increment is the smallest for which the largest magnitude of the array fits the type, as
    ``"auto"`` chooses it. An axis that is incremental or of a floating point type is left alone.
    """
    arrays = {"x": x, "y": y, "z": z}
    axes = {}
    for name, array in arrays.items():
        axis: Axis = getattr(header, name)
        info = get_data_type(axis.data_type)
        if array is None or axis.is_incremental or not info.is_integer:
            axes[name] = axis
        else:
            axes[name] = replace(axis, increment=suggest_scale(array, info))
    return replace(header, **axes)


def _middle(stored: np.ndarray) -> float:
    """Return the middle of the range of the finite values, or zero if there are none."""
    finite = stored[np.isfinite(stored)]
    if finite.size == 0:
        return 0.0
    return (float(finite.min()) + float(finite.max())) / 2.0
