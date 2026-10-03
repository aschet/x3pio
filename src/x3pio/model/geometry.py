# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Coordinate systems and the placement of the view system in the global one.

An x3p file stores its coordinates in its own system, the view coordinate system: the system in
which the instrument or the software described the points. The file also says where that system lies
in the global coordinate system, the one the data set belongs to, by a rotation and an offset. Both
systems are right-handed, and a pure rotation keeps that. The standard writes this as one formula
for the stored values ``s`` of the three axes, their increments ``I`` and the offsets ``O``::

    global = R @ (I * s) + O

``I * s`` is what this library calls the view coordinates, in metres, and ``R`` and ``O`` are
what :class:`Placement` holds. Nothing in a file is transformed implicitly: the stored
coordinates stay as they are, and a :class:`CoordinateSystem` says which of the two systems a
function returns or expects.

Without a placement the two systems are the same. ``points(coordinate_system=...)`` of a layer
returns its points in either system, ``with_placement`` and ``transformed`` change where the
points lie in the global system, ``globalized`` makes the global coordinates the stored ones, and
``centered`` centres the stored coordinates on the origin, as the standard recommends. To write a
file with a placement, pass ``placement=`` and ``coordinate_system=`` to the constructors;
``coordinate_system`` says which system the values you give are in.
"""

from __future__ import annotations

import math
from enum import Enum, auto

import numpy as np
from numpy.typing import ArrayLike

from ..exceptions import X3pFormatError

__all__ = ["CoordinateSystem", "Placement", "validate_rotation"]

#: Largest deviation from a pure rotation matrix that :func:`validate_rotation` accepts.
_ROTATION_TOLERANCE = 1e-6


class CoordinateSystem(Enum):
    """The coordinate system that coordinates are given in, or are asked for.

    The same points have different coordinates in the two systems, unless the placement of the
    file is the identity.
    """

    #: The system the file stores its coordinates in: the stored values times their increments.
    VIEW = auto()
    #: The system the data set belongs to: the view system turned and moved by the placement.
    GLOBAL = auto()


def validate_rotation(rotation: np.ndarray) -> None:
    """Reject a matrix that is not a pure 3x3 rotation.

    The standard allows only a rotation: no mirroring, scaling or shearing.
    The matrix must be orthonormal with a determinant of ``+1``, within a
    small tolerance, and each element must lie within ``[-1, 1]``.

    :param rotation: The matrix to check.
    :raises X3pFormatError: If ``rotation`` is not such a matrix.
    """
    if rotation.shape != (3, 3):
        raise X3pFormatError(f"Rotation must be a 3x3 matrix, got shape {rotation.shape}")
    if not np.all(np.isfinite(rotation)) or np.abs(rotation).max() > 1.0:
        raise X3pFormatError("Rotation elements must lie within [-1, 1]")
    orthogonal = np.allclose(rotation @ rotation.T, np.eye(3), atol=_ROTATION_TOLERANCE)
    if not orthogonal or abs(np.linalg.det(rotation) - 1.0) > _ROTATION_TOLERANCE:
        raise X3pFormatError("Rotation must be a pure rotation matrix (orthonormal, determinant 1)")


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class Placement:
    """The pose of the view coordinate system in the global one: a rotation and an offset.

    This is the rotation matrix and the offset of the standard, which together make up its
    coordinate transformation, without the increments: they scale the stored numbers and belong
    to the axes of the file.

    A point with the view coordinates ``p`` has the global coordinates ``rotation @ p +
    offset``. A placement is immutable. Placements compose with ``@``, the way matrices do:
    ``(a @ b).apply(p)`` is ``a.apply(b.apply(p))``.

    The offset of an x3p file is the global position of the origin of the view system. For an
    axis that is incremental, whose coordinates are the position of the point in the matrix
    times the increment, the origin is the first point.

    >>> import numpy as np
    >>> import x3pio
    >>> quarter_turn = x3pio.Placement.from_axis_angle([0, 0, 1], np.pi / 2)
    >>> p = x3pio.Placement.translation(1.0, 0.0, 0.0) @ quarter_turn
    >>> p.apply([[1.0, 0.0, 0.0]]).round(12).tolist()
    [[1.0, 1.0, 0.0]]
    >>> p.inverse().apply(p.apply([[1.0, 2.0, 3.0]])).round(12).tolist()
    [[1.0, 2.0, 3.0]]

    :param rotation: The ``3 x 3`` rotation of the view system into the global one, the
        identity by default. It must be a pure rotation, without mirroring or scaling.
    :param offset: The offsets of the x, y and z axis in metres, zero by default.
    :raises X3pFormatError: If the rotation is not a rotation, or the offset is not three finite
        numbers.
    """

    __slots__ = ("_offset", "_rotation")

    def __init__(self, rotation: ArrayLike | None = None, offset: ArrayLike | None = None) -> None:
        """Make a placement; see the class for the parameters."""
        matrix = np.eye(3) if rotation is None else np.array(rotation, dtype=np.float64)
        shifts = np.zeros(3) if offset is None else np.array(offset, dtype=np.float64)
        validate_rotation(matrix)
        if shifts.shape != (3,) or not np.all(np.isfinite(shifts)):
            raise X3pFormatError("The offset must be three finite numbers: x, y and z")
        self._rotation = _frozen(matrix)
        self._offset = _frozen(shifts)

    @classmethod
    def _unchecked(cls, rotation: np.ndarray, offset: np.ndarray) -> Placement:
        """Make a placement without checking the rotation, for what a file holds.

        Reading tolerates a rotation that is slightly off; writing refuses it.
        """
        placement = cls.__new__(cls)
        placement._rotation = _frozen(np.array(rotation, dtype=np.float64))
        placement._offset = _frozen(np.array(offset, dtype=np.float64))
        return placement

    @classmethod
    def identity(cls) -> Placement:
        """Return the placement that leaves the coordinates as they are."""
        return cls()

    @classmethod
    def translation(cls, x: float, y: float, z: float) -> Placement:
        """Return a placement that only moves, by ``x``, ``y`` and ``z`` in metres."""
        return cls(offset=[x, y, z])

    @classmethod
    def from_axis_angle(cls, axis: ArrayLike, angle: float) -> Placement:
        """Return a placement that only turns, by ``angle`` radians about ``axis``.

        The turn is counterclockwise when the axis points at the viewer.

        :param axis: The direction of the axis, three numbers that need not be of length one.
        :param angle: The angle in radians.
        :raises X3pFormatError: If the axis is not three finite numbers that are not all zero.
        """
        direction = np.array(axis, dtype=np.float64)
        length = float(np.linalg.norm(direction)) if direction.shape == (3,) else math.nan
        if not math.isfinite(length) or length == 0.0:
            raise X3pFormatError("The axis must be three finite numbers that are not all zero")
        ux, uy, uz = direction / length
        cross = np.array([[0.0, -uz, uy], [uz, 0.0, -ux], [-uy, ux, 0.0]])
        rotation = (
            math.cos(angle) * np.eye(3)
            + math.sin(angle) * cross
            + (1.0 - math.cos(angle)) * np.outer(direction / length, direction / length)
        )
        return cls(rotation=rotation)

    @property
    def rotation(self) -> np.ndarray:
        """The ``3 x 3`` rotation matrix, read-only."""
        return self._rotation

    @property
    def offset(self) -> np.ndarray:
        """The offsets of the x, y and z axis in metres, read-only."""
        return self._offset

    @property
    def has_rotation(self) -> bool:
        """Whether the placement turns anything, which the identity rotation does not."""
        return not np.array_equal(self._rotation, np.eye(3))

    @property
    def is_identity(self) -> bool:
        """Whether the placement leaves the coordinates as they are."""
        return not self.has_rotation and not self._offset.any()

    def apply(self, points: ArrayLike) -> np.ndarray:
        """Turn view coordinates into global ones: ``rotation @ point + offset`` for each point.

        :param points: View coordinates in metres, shaped ``(..., 3)``. A point with ``NaN`` in
            any coordinate has ``NaN`` in all three of its result if the placement has a
            rotation, since the rotation mixes them; without a rotation, the others stay known.
        :returns: The global coordinates, shaped like ``points``.
        """
        array = np.asarray(points, dtype=np.float64)
        if self.has_rotation:
            with np.errstate(invalid="ignore"):
                array = np.asarray(array @ self._rotation.T)
        return np.asarray(array + self._offset)

    def inverse(self) -> Placement:
        """Return the placement that undoes this one: global coordinates into view ones."""
        rotation = self._rotation.T
        return Placement._unchecked(rotation, -(rotation @ self._offset))

    def __matmul__(self, other: Placement) -> Placement:
        """Return the placement that applies ``other`` first and then this one."""
        if not isinstance(other, Placement):
            return NotImplemented
        return Placement._unchecked(
            self._rotation @ other._rotation, self._rotation @ other._offset + self._offset
        )

    def __eq__(self, other: object) -> bool:
        """Compare the rotation and the offset exactly."""
        if not isinstance(other, Placement):
            return NotImplemented
        return bool(
            np.array_equal(self._rotation, other._rotation)
            and np.array_equal(self._offset, other._offset)
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Show the rotation and the offset."""
        return f"Placement(rotation={self._rotation.tolist()}, offset={self._offset.tolist()})"
