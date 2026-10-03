# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Layers of profiles and surfaces.

A profile or a surface of x3p can have several layers, which lie next to each other: the points
keep the neighbours of their place in the layer, and each point also has the points at the same
place of the adjacent layers as neighbours. Most files have a single layer. A point cloud has no
layers in the format; x3pio presents its list of points as one layer.

A layer is a complete topography: it holds the arrays of its points together with the axes and the
placement of the file, so it gives its points without the file. A file keeps its layers: asking
for them again gives the same objects, and layers are equal if their axes, placement and values
are.

The axes and the placement of a layer are immutable values, and its arrays can be written to. To
make a layer of other data, such as a cropped array or other points, make it with ``from_array``
or ``from_points`` and give it to a file with ``X3pFile.with_layers``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, Self, TypeVar

import numpy as np

from ..exceptions import X3pFormatError
from .arithmetic import view_points
from .geometry import CoordinateSystem, Placement
from .header import Header

if TYPE_CHECKING:
    from typing import Unpack

    from numpy.typing import ArrayLike

    from .files import ArrayOptions, PointOptions

__all__ = [
    "Layer",
    "Point",
    "PointLayer",
    "ProfileLayer",
    "SurfaceLayer",
]


_LayerT = TypeVar("_LayerT", bound="Layer")


class Point(NamedTuple):
    """One 3-D point; ``NaN`` for a coordinate that is not measured."""

    x: float
    y: float
    z: float


class Layer:
    """A layer of a profile, a surface or a point cloud: the data and the geometry that all share.

    A layer knows its axes and its placement, so it can give its points without the file. The
    axes and the placement are immutable values; the arrays can be written to, see :attr:`z`.
    Two layers are equal if they are of the same kind and have equal axes, placement and values,
    ``NaN`` equal to ``NaN``. Since the values can change, a layer is not hashable.

    :param z: The heights, shaped like the layer; ``NaN`` for a point that is not measured.
    :param x: The x coordinates if the x axis is absolute, otherwise ``None``.
    :param y: The y coordinates if the y axis is absolute, otherwise ``None``.
    :param header: The axes of the file, with their increments.
    :param placement: Where the view coordinate system lies in the global one; the identity by
        default.
    """

    def __init__(
        self,
        z: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        header: Header,
        placement: Placement | None = None,
    ) -> None:
        """Make a layer of arrays, which it keeps as they are."""
        self._z = z
        self._x = x
        self._y = y
        self._header = header
        self._placement = Placement.identity() if placement is None else placement

    @property
    def header(self) -> Header:
        """The axes of the layer, with their increments; an immutable value."""
        return self._header

    @property
    def placement(self) -> Placement:
        """Where the view coordinate system lies in the global one; an immutable value."""
        return self._placement

    @property
    def z(self) -> np.ndarray:
        """The heights in the stored shape; ``NaN`` for a point that is not measured.

        The array can be written to: ``layer.z[0, 1] = 1.0``, ``layer.z -= 1.0`` and
        ``layer.z = other`` all change the file that holds the layer. Assigning fills the array
        and so needs a value of the same shape, or a single number; it does not replace the
        array. A layer of a copy that shares its arrays is read-only; :meth:`X3pFile.copy` makes
        a writable copy.

        :raises ValueError: On assigning a value of another shape, or to a read-only layer.
        """
        return self._z

    @z.setter
    def z(self, value: ArrayLike) -> None:
        _fill(self._z, value, "z")

    @property
    def x(self) -> np.ndarray | None:
        """The stored x coordinates in metres if the x axis is absolute, otherwise ``None``.

        With an incremental axis the coordinates follow from the place of a point, see
        ``x_values``, and there is nothing to write to. Assigning works like for :attr:`z`.

        :raises X3pFormatError: On assigning to a layer whose x axis is incremental; make a layer
            from the coordinates of its points, with ``from_points``.
        """
        return self._x

    @x.setter
    def x(self, value: ArrayLike) -> None:
        _fill(_stored(self._x, "x"), value, "x")

    @property
    def y(self) -> np.ndarray | None:
        """The stored y coordinates in metres if the y axis is absolute, otherwise ``None``.

        See :attr:`x`.
        """
        return self._y

    @y.setter
    def y(self, value: ArrayLike) -> None:
        _fill(_stored(self._y, "y"), value, "y")

    def __eq__(self, other: object) -> bool:
        """Compare the kind, axes, placement and values of two layers."""
        if not isinstance(other, Layer):
            return NotImplemented
        return (
            type(self) is type(other)
            and (self.header is other.header or self.header == other.header)
            and (self.placement is other.placement or self.placement == other.placement)
            and _same_array(self.z, other.z)
            and _same_optional_array(self.x, other.x)
            and _same_optional_array(self.y, other.y)
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Show the arrays, the axes and the placement."""
        return (
            f"{type(self).__name__}(z={self.z!r}, x={self.x!r}, y={self.y!r}, "
            f"header={self.header!r}, placement={self.placement!r})"
        )

    def _derived(
        self,
        *,
        header: Header | None = None,
        placement: Placement | None = None,
        arrays: Literal["copy", "read_only"] | None = None,
    ) -> Self:
        """Return a layer of the same kind, with other axes or placement and other arrays.

        ``arrays`` makes copies of the arrays, or read-only views of them; without it the new
        layer has the same arrays.
        """

        def make(array: np.ndarray | None) -> np.ndarray | None:
            if array is None or arrays is None:
                return array
            return array.copy() if arrays == "copy" else _read_only(array)

        return type(self)(
            make(self._z),  # type: ignore[arg-type]
            make(self._x),
            make(self._y),
            header=self._header if header is None else header,
            placement=self._placement if placement is None else placement,
        )

    def _cube(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``z``, ``x`` and ``y`` shaped as ``view_points`` takes them."""
        return self.z, self.x, self.y

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the layer."""
        return tuple(self.z.shape)

    @property
    def valid(self) -> np.ndarray:
        """Boolean mask of the points that are measured."""
        return ~np.isnan(self.z)

    @property
    def is_regular_grid(self) -> bool:
        """Whether the x and y axes are both incremental, so the points lie on a regular grid.

        The grid is then described completely by ``x_values``, ``y_values``, ``spacing`` and
        ``extent`` of the layer. It is never so for a point cloud, whose axes are absolute, and not
        for a profile or a surface with a stored x or y coordinate for each point.
        """
        return self.header.x.is_incremental and self.header.y.is_incremental

    def points(
        self, *, coordinate_system: CoordinateSystem = CoordinateSystem.GLOBAL
    ) -> np.ndarray:
        """Return the points as an array of ``x``, ``y``, ``z`` in metres, shaped like the layer.

        The shape is ``z.shape + (3,)``: ``(rows, columns, 3)`` for a surface layer,
        ``(points, 3)`` for a profile layer and for a point cloud. A point that is not measured
        has ``NaN`` in ``z``, and in ``x`` and ``y`` too if the placement turns the points, since
        the rotation couples them. The ``from_points`` constructors take this shape.

        :param coordinate_system: :attr:`CoordinateSystem.GLOBAL`, with the placement applied,
            which is the default, or :attr:`CoordinateSystem.VIEW`, as stored.
        """
        view = view_points(self.header, *self._cube())
        if coordinate_system is CoordinateSystem.GLOBAL:
            view = self.placement.apply(view)
        return view.reshape(*self.z.shape, 3)

    def iter_points(
        self,
        *,
        coordinate_system: CoordinateSystem = CoordinateSystem.GLOBAL,
        drop_invalid: bool = False,
    ) -> Iterator[Point]:
        """Iterate over the points of :meth:`points` in storage order.

        :param coordinate_system: See :meth:`points`.
        :param drop_invalid: Leave out the points that are not measured.
        :returns: The points as :class:`Point` tuples.
        """
        points = self.points(coordinate_system=coordinate_system).reshape(-1, 3)
        if drop_invalid:
            points = points[np.isfinite(points).all(axis=1)]
        for x, y, z in points.tolist():
            yield Point(x, y, z)


class ProfileLayer(Layer):
    """One layer of a :class:`~x3pio.Profile`.

    Writing to the arrays changes the file that holds the layer. ``z``, ``x`` and ``y`` are
    shaped ``(points,)``.
    """

    def _cube(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        def cube(array: np.ndarray | None) -> np.ndarray | None:
            return None if array is None else array.reshape(1, 1, -1)

        return cube(self.z), cube(self.x), cube(self.y)  # type: ignore[return-value]

    @property
    def x_values(self) -> np.ndarray:
        """X coordinates of the points in metres: ``x_u = (u - 1) * increment``.

        :raises ValueError: If the x axis is absolute.
        """
        if not self.header.x.is_incremental:
            raise ValueError("x_values needs an incremental x axis")
        return np.arange(self.z.shape[-1]) * self.header.x.increment

    @property
    def spacing(self) -> float:
        """The distance between neighbouring points along x in metres: the x increment.

        :raises ValueError: If the x axis is absolute.
        """
        if not self.header.x.is_incremental:
            raise ValueError("spacing needs an incremental x axis")
        return self.header.x.increment

    @classmethod
    def from_array(
        cls, data: ArrayLike, *, x_scale: float, **options: Unpack[ArrayOptions]
    ) -> ProfileLayer:
        """Make a layer alone from an array of heights, see :meth:`Profile.from_array`.

        >>> import numpy as np
        >>> import x3pio
        >>> layer = x3pio.ProfileLayer.from_array(np.zeros(4), x_scale=1e-6)
        >>> layer.points().shape
        (4, 3)
        """
        from .files import Profile

        return _one(
            Profile.from_array(
                np.asarray(data, dtype=np.float64), x_scale=x_scale, **options
            ).layers
        )

    @classmethod
    def from_points(cls, points: ArrayLike, **options: Unpack[PointOptions]) -> ProfileLayer:
        """Make a layer alone from its points, see :meth:`Profile.from_points`."""
        from .files import Profile

        return _one(Profile.from_points(points, **options).layers)


class SurfaceLayer(Layer):
    """One layer of a :class:`~x3pio.Surface`.

    Writing to the arrays changes the file that holds the layer. ``z``, ``x`` and ``y`` are
    shaped ``(rows, columns)``.

    Make a layer of heights on a regular grid. ``spacing`` is the distance between the points and
    ``extent`` the cell edges in the form that the ``extent`` argument of an image plot takes;
    ``y`` increases with the row index::

        >>> import numpy as np
        >>> import x3pio
        >>> heights = np.array([[0.0, 1.0, np.nan], [2.0, 3.0, 4.0]]) * 1e-6
        >>> layer = x3pio.SurfaceLayer.from_array(heights, x_scale=2e-6, y_scale=1e-6)
        >>> layer.spacing, layer.valid.tolist()
        ((2e-06, 1e-06), [[True, True, False], [True, True, True]])
        >>> [round(value * 1e6, 3) for value in layer.extent]
        [-1.0, 5.0, -0.5, 1.5]
        >>> layer.points().shape
        (2, 3, 3)
    """

    def _cube(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        return (
            self.z[None],
            None if self.x is None else self.x[None],
            None if self.y is None else self.y[None],
        )

    @property
    def x_values(self) -> np.ndarray:
        """X coordinates of the columns in metres: ``x_u = (u - 1) * increment``.

        :raises ValueError: If the x axis is absolute.
        """
        if not self.header.x.is_incremental:
            raise ValueError("x_values needs an incremental x axis")
        return np.arange(self.z.shape[-1]) * self.header.x.increment

    @property
    def y_values(self) -> np.ndarray:
        """Y coordinates of the rows in metres: ``y_v = (v - 1) * increment``.

        ``y`` increases with the row index.

        :raises ValueError: If the y axis is absolute.
        """
        if not self.header.y.is_incremental:
            raise ValueError("y_values needs an incremental y axis")
        return np.arange(self.z.shape[-2]) * self.header.y.increment

    @property
    def spacing(self) -> tuple[float, float]:
        """The distance between neighbouring points along x and along y in metres.

        These are the increments of the axes, so nobody has to multiply them by the size.

        :raises ValueError: If the x or the y axis is absolute.
        """
        if not (self.header.x.is_incremental and self.header.y.is_incremental):
            raise ValueError("spacing needs incremental x and y axes")
        return self.header.x.increment, self.header.y.increment

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """The edges of the grid cells as ``(x0, x1, y0, y1)`` in metres, in the view system.

        A point lies in the middle of its cell, so the extent reaches half a step beyond the
        first and the last point: for ``x_values`` it is ``-dx / 2`` to ``x_values[-1] + dx / 2``.
        This is what the ``extent`` of an image plot of the heights takes, with the origin in
        the lower left corner, since ``y`` grows with the row::

            plt.imshow(layer.z, origin="lower", extent=layer.extent)

        :raises ValueError: If the x or the y axis is absolute.
        """
        dx, dy = self.spacing
        rows, columns = self.z.shape
        return -dx / 2, (columns - 0.5) * dx, -dy / 2, (rows - 0.5) * dy

    @classmethod
    def from_array(
        cls, data: ArrayLike, *, x_scale: float, y_scale: float, **options: Unpack[ArrayOptions]
    ) -> SurfaceLayer:
        """Make a layer alone from an array of heights, see :meth:`Surface.from_array`.

        >>> import numpy as np
        >>> import x3pio
        >>> layer = x3pio.SurfaceLayer.from_array(np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
        >>> layer.points().shape
        (2, 3, 3)
        """
        from .files import Surface

        return _one(
            Surface.from_array(
                np.asarray(data, dtype=np.float64), x_scale=x_scale, y_scale=y_scale, **options
            ).layers
        )

    @classmethod
    def from_points(cls, points: ArrayLike, **options: Unpack[PointOptions]) -> SurfaceLayer:
        """Make a layer alone from its points, see :meth:`Surface.from_points`."""
        from .files import Surface

        return _one(Surface.from_points(points, **options).layers)


class PointLayer(Layer):
    """The points of a :class:`~x3pio.PointCloud`, as the one layer that it presents.

    The format has no layers for a point cloud. x3pio presents its list of points as a single
    layer, so that every file is used the same way. ``z``, ``x`` and ``y`` are shaped
    ``(points,)`` and all three axes are absolute.
    """

    @classmethod
    def from_points(cls, points: ArrayLike, **options: Unpack[PointOptions]) -> PointLayer:
        """Make a layer alone from an ``(N, 3)`` array, see :meth:`PointCloud.from_points`.

        >>> import numpy as np
        >>> import x3pio
        >>> layer = x3pio.PointLayer.from_points(np.array([[0.0, 1.0, 2.0]]))
        >>> layer.points().tolist()
        [[0.0, 1.0, 2.0]]
        """
        from .files import PointCloud

        return PointCloud.from_points(points, **options).layer


def _one(layers: Sequence[_LayerT]) -> _LayerT:
    """Return the layer of a file made for a layer alone, which is refused if it has several."""
    if len(layers) != 1:
        raise ValueError(f"A layer is made from the data of one layer, not {len(layers)}")
    return layers[0]


def _same_array(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.array_equal(first, second, equal_nan=True))


def _same_optional_array(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    if first is None or second is None:
        return first is second
    return _same_array(first, second)


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a view of ``array`` that shares its memory, but cannot be written to."""
    view = array.view()
    view.flags.writeable = False
    return view


def _stored(array: np.ndarray | None, name: str) -> np.ndarray:
    """Return the stored coordinates of an axis, or refuse if the axis stores none."""
    if array is None:
        raise X3pFormatError(
            f"The {name} axis is incremental, so the layer stores no {name}; "
            "make a layer with the coordinates of its points, with from_points"
        )
    return array


def _fill(target: np.ndarray, value: ArrayLike, name: str) -> None:
    """Write ``value`` into ``target``, a number for all or an array of the same shape."""
    if value is target:  # ``layer.z -= 1`` assigns the array that it has changed in place
        return
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in ((), target.shape):
        raise ValueError(f"{name} has the shape {target.shape}, not {array.shape}")
    if not target.flags.writeable:
        raise ValueError(
            f"The {name} of this layer is read-only, as it shares its arrays with another file; "
            "make a writable file with copy()"
        )
    target[...] = array
