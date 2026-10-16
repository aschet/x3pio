# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""File types: profile, surface and point cloud.

An x3p file holds the 3-D coordinates of measured points in one of three shapes. The standard
calls them feature types, and each is a type here:

- A :class:`Profile` is a line of points in a given order. The points need not lie on a straight
  line; each has the points before and after it as neighbours.
- A :class:`Surface` is a data matrix: a matrix of points with a neighbourhood relation, so
  that the neighbours of a point in the matrix are its neighbours in space. The points can lie
  on a regular grid, but do not have to.
- A :class:`PointCloud` is a list of points without any order or neighbourhood.

Each axis of a file is either incremental or absolute. The coordinate along an incremental
axis is the index of the point in the matrix times the increment: a regular grid, with nothing
stored for the axis. An absolute axis stores a coordinate for each point, as a number that is
multiplied by the increment to give metres. The x and y axes of a profile and of a surface can be
of either kind, the z axis, the height, is always absolute, and all three axes of a point cloud
are absolute.

A point of a data matrix has the indices ``u``, ``v`` and ``w``: the column, the row and the
layer. The matrix has ``SizeX``, ``SizeY`` and ``SizeZ`` points along them, and the points are
stored with ``u`` varying fastest, then ``v``, then ``w``. The arrays of a layer are therefore
shaped ``(rows, columns)``. Only an incremental axis follows the index: ``x`` from ``u`` and ``y``
from ``v``. The coordinates of an absolute axis are independent of the index.

Profiles and surfaces can have several layers (the standard speaks of multilayer data), which lie
next to each other, so that each point also has the points at the same place in the adjacent
layers as neighbours. Most files have a single layer.

The data of every file is reached the same way, through its ``layers``: each layer holds its
``z``, ``x`` and ``y`` arrays and gives its ``points``. A point cloud presents its list of points
as one layer, since the format has no layers for it. The layers share the axes and the placement.
The file itself holds the layers and what belongs to the whole file: the revision of the standard,
the metadata and the vendor extensions.

The stored coordinates, in metres, are in the view coordinate system of the file, and a
:class:`~x3pio.Placement` says where that system lies in the global one. Only
the instrument or the software that wrote the file knows what the global system is.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import IO, Any, ClassVar, TypeVar, Unpack

import numpy as np
from numpy.typing import ArrayLike

from ..exceptions import X3pFormatError
from .arithmetic import resolve_coordinate_system
from .base import X3pFile
from .datatypes import (
    DataType,
    get_data_type,
    suggest_scale,
)
from .geometry import Placement
from .header import (
    DEFAULT_REVISION,
    Axis,
    AxisType,
    FeatureType,
    Header,
)
from .layers import Layer, PointLayer, ProfileLayer, SurfaceLayer
from .options import ArrayOptions, PointOptions, _check_options, _new_metadata
from .storage import DataStorage

__all__ = [
    "PointCloud",
    "Profile",
    "Surface",
]


def _matrix_file(
    cls: type[Profile] | type[Surface],
    matrix: np.ndarray,
    flat: np.ndarray,
    *,
    x_scale: float,
    y_scale: float,
    options: ArrayOptions,
) -> Any:
    """Make a profile or a surface of ``matrix``, the heights as ``(layers, rows, columns)``.

    ``flat`` is the same array in the shape that ``cls`` holds, as a view.
    """
    _check_options(options, ArrayOptions.__annotations__)
    placement, is_global = resolve_coordinate_system(
        options.get("placement"), options.get("coordinate_system")
    )
    if is_global and placement.has_rotation:
        raise X3pFormatError(
            "The global heights of a rotated grid are not on a regular grid; "
            "give the heights of the view system with coordinate_system=CoordinateSystem.VIEW"
        )
    if is_global:
        matrix = matrix - placement.offset[2]
        flat = matrix.reshape(flat.shape)
    resolved_type = get_data_type(options.get("z_type", DataType.FLOAT64))
    z_scale = options.get("z_scale", "auto")
    increment = suggest_scale(matrix, resolved_type.type) if z_scale == "auto" else z_scale
    header = Header(
        x=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, x_scale),
        y=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, y_scale),
        z=Axis(AxisType.ABSOLUTE, resolved_type.type, increment),
    )
    metadata = options.get("metadata")
    file = cls._from_stacked(
        header,
        flat,
        placement=placement,
        revision=options.get("revision", DEFAULT_REVISION),
        metadata=_new_metadata(metadata),
        storage=options.get("storage", DataStorage.BINARY),
    )
    file._validate()
    return file


def _absolute_file(cls: type[X3pFile], points: np.ndarray, options: PointOptions) -> Any:
    """Make a file of ``cls`` whose axes are all absolute, from points shaped ``(..., 3)``.

    The last axis of ``points`` holds ``x``, ``y`` and ``z``; the others are the shape of the
    file, which the three arrays of the file get as well.
    """
    _check_options(options, PointOptions.__annotations__)
    placement, is_global = resolve_coordinate_system(
        options.get("placement"), options.get("coordinate_system")
    )
    if is_global:
        points = placement.inverse().apply(points)
    resolved_type = get_data_type(options.get("data_type", DataType.FLOAT64))
    scale = options.get("scale", "auto")
    axes = [
        Axis(
            AxisType.ABSOLUTE,
            resolved_type.type,
            suggest_scale(points[..., index], resolved_type.type) if scale == "auto" else scale,
        )
        for index in range(3)
    ]
    header = Header(x=axes[0], y=axes[1], z=axes[2])
    metadata = options.get("metadata")
    file = cls._from_stacked(
        header,
        points[..., 2].copy(),
        points[..., 0].copy(),
        points[..., 1].copy(),
        placement=placement,
        revision=options.get("revision", DEFAULT_REVISION),
        metadata=_new_metadata(metadata),
        storage=options.get("storage", DataStorage.BINARY),
    )
    file._validate()
    return file


def _layer_arrays(layers: Sequence[Any], ndim: int, name: str) -> list[np.ndarray]:
    """Return the heights of layers that are arrays, checked to fit together."""
    if len(layers) == 0:
        raise X3pFormatError(f"A {name} needs at least one layer")
    arrays: list[np.ndarray] = []
    for layer in layers:
        if isinstance(layer, Layer):
            raise X3pFormatError(
                f"Give the layers of a {name} either all as arrays or all as layers"
            )
        array = np.asarray(layer, dtype=np.float64)
        if array.ndim != ndim:
            raise X3pFormatError(
                f"Each layer of a {name} is a {ndim}-D array of heights, got {array.ndim}-D"
            )
        arrays.append(array)
    if any(array.shape != arrays[0].shape for array in arrays):
        raise X3pFormatError(f"The layers of a {name} must have the same shape")
    return arrays


_LayerT = TypeVar("_LayerT", bound=Layer)


def _file_of_layers(
    cls: type[Profile] | type[Surface], layers: Sequence[Any], options: ArrayOptions
) -> Any:
    """Make a profile or a surface of copies of layers, which carry their axes and placement."""
    cls._check_layers(layers)
    _check_options(options, {"metadata": 0, "storage": 0, "revision": 0})  # the rest is in layers
    metadata = options.get("metadata")
    file = cls(
        [layer._derived(arrays="copy") for layer in layers],
        revision=options.get("revision", DEFAULT_REVISION),
        metadata=_new_metadata(metadata),
        storage=options.get("storage", DataStorage.BINARY),
    )
    file._validate()
    return file


class Profile(X3pFile):
    """A profile: a linear sequence of points, in one or several layers.

    The data is in :attr:`layers`, one :class:`ProfileLayer` for each layer, whose ``z`` is
    shaped ``(points,)``. The y axis of a profile has a single row, so ``y`` does not vary along a
    regular profile. The points need not lie on a straight line.

    Make a profile from heights at equal steps along x::

        >>> import numpy as np
        >>> import x3pio
        >>> scan = x3pio.Profile.from_array(np.sin(np.linspace(0, 6, 100)) * 1e-6, x_scale=1e-6)
        >>> len(scan.layers), scan.layers[0].z.shape
        (1, (100,))
        >>> scan.layers[0].x_values[:2].tolist()
        [0.0, 1e-06]

    Several scans of the same line, as the layers of one file::

        >>> profile = x3pio.Profile.from_layers([np.zeros(50), np.ones(50)], x_scale=1e-6)
        >>> [float(layer.z[0]) for layer in profile.layers]
        [0.0, 1.0]

    A profile that follows an arbitrary path, from the coordinates of its points; each layer gives
    its points in the same shape again, :meth:`~ProfileLayer.points`::

        >>> path = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 3.0]]) * 1e-3
        >>> curved = x3pio.Profile.from_points(path)
        >>> curved.layers[0].points().shape, curved.layers[0].x.tolist()
        ((3, 3), [0.0, 0.001, 0.002])
    """

    _FEATURE: ClassVar[FeatureType] = FeatureType.PROFILE
    _NAME: ClassVar[str] = "profile"
    _LAYER: ClassVar[type[Layer]] = ProfileLayer
    _LAYER_NDIM: ClassVar[int] = 1
    _layers: tuple[ProfileLayer, ...]

    def _cube(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        def cube(array: np.ndarray | None) -> np.ndarray | None:
            return None if array is None else array.reshape(array.shape[0], 1, array.shape[1])

        z, x, y = self._stacked()
        return cube(z), cube(x), cube(y)  # type: ignore[return-value]

    def _uncube(
        self, z: np.ndarray, x: np.ndarray | None, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        def flat(array: np.ndarray | None) -> np.ndarray | None:
            return None if array is None else array.reshape(array.shape[0], array.shape[2])

        return flat(z), flat(x), flat(y)  # type: ignore[return-value]

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        x_scale: float,
        **options: Unpack[ArrayOptions],
    ) -> Profile:
        """Build a profile from an array of heights.

        The array is used as it is when it is already ``float64``, not copied, so do not
        change it while the file is in use.

        :param data: Heights in metres: ``(points,)`` for a profile of one layer, or
            ``(layers, points)``. ``NaN`` marks a point that is not measured.
        :param x_scale: Sampling interval along x, in metres. A profile has a single row, so its
            y axis, which it does not use, gets the same increment.
        :param options: See :class:`ArrayOptions`.
        :raises X3pFormatError: If ``data`` is not one- or two-dimensional, a scale is not
            positive, or global heights are given with a rotated placement.
        :raises TypeError: If ``placement`` is given without ``coordinate_system``.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Profile.from_array(np.zeros((2, 10)), x_scale=1e-6)
        >>> len(file.layers), file.layers[0].z.shape
        (2, (10,))
        """
        array = np.asarray(data, dtype=np.float64)
        if array.ndim not in (1, 2):
            raise X3pFormatError("Data must be a profile, (points,), or layers, (layers, points)")
        layers = array.reshape(-1, array.shape[-1])
        return _matrix_file(  # type: ignore[no-any-return]
            cls,
            layers.reshape(layers.shape[0], 1, layers.shape[1]),
            layers,
            x_scale=x_scale,
            y_scale=x_scale,
            options=options,
        )

    @classmethod
    def from_layers(
        cls,
        layers: Iterable[ArrayLike | ProfileLayer],
        *,
        x_scale: float | None = None,
        **options: Unpack[ArrayOptions],
    ) -> Profile:
        """Build a profile of several layers, from arrays of heights or from layers.

        The layers have to have the same number of points. Given as arrays, each is one layer of
        heights and ``x_scale`` says how far apart the points are. Given as :class:`ProfileLayer`,
        they bring their axes and their placement, which have to agree, and no scale is given.

        :param layers: Arrays of heights shaped ``(points,)`` in metres, with ``NaN`` for a point
            that is not measured, or layers, for example from other files.
        :param x_scale: Sampling interval along x, in metres, for arrays.
        :param options: See :class:`ArrayOptions`; with layers only ``metadata``, ``storage`` and
            ``revision`` apply.
        :raises X3pFormatError: If there is no layer, arrays are not one-dimensional or layers
            differ in their number of points, their axes or their placement.
        :raises TypeError: If ``x_scale`` is left out for arrays, or given with layers.

        >>> import numpy as np
        >>> import x3pio
        >>> first, second = np.zeros(5), np.ones(5)
        >>> file = x3pio.Profile.from_layers([first, second], x_scale=1e-6)
        >>> len(file.layers), file.layers[1].z.tolist()
        (2, [1.0, 1.0, 1.0, 1.0, 1.0])
        >>> again = x3pio.Profile.from_layers(file.layers)
        >>> again.layers[1].z.tolist()
        [1.0, 1.0, 1.0, 1.0, 1.0]
        """
        items = tuple(layers)
        if len(items) > 0 and all(isinstance(layer, ProfileLayer) for layer in items):
            if x_scale is not None:
                raise TypeError("Layers bring their scale; leave x_scale out")
            return _file_of_layers(cls, items, options)  # type: ignore[no-any-return]
        arrays = _layer_arrays(items, 1, "profile")
        if x_scale is None:
            raise TypeError("x_scale is required for arrays of heights")
        return cls.from_array(np.stack(arrays), x_scale=x_scale, **options)

    @classmethod
    def from_points(cls, points: ArrayLike, **options: Unpack[PointOptions]) -> Profile:
        """Build a profile from the coordinates of its points, which may follow any path.

        All three axes are absolute: every point stores its ``x``, ``y`` and ``z``, in the order
        along the profile. Use :meth:`from_array` for a profile along a straight line at equal
        steps.

        :param points: ``(points, 3)`` array of ``x``, ``y``, ``z`` in metres for a profile of one
            layer, or ``(layers, points, 3)``. ``NaN`` in ``z`` marks a point that is not measured.
        :param options: See :class:`PointOptions`.
        :raises X3pFormatError: If ``points`` has another shape, does not fit ``data_type``
            or does not fit ``data_type`` without overflow.
        :raises TypeError: If ``placement`` is given without ``coordinate_system``.

        >>> import numpy as np
        >>> import x3pio
        >>> path = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 3.0]])
        >>> file = x3pio.Profile.from_points(path)
        >>> file.layers[0].z.shape, file.layers[0].x.tolist()
        ((3,), [0.0, 1.0, 2.0])
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim not in (2, 3) or array.shape[-1] != 3:
            raise X3pFormatError(
                "Points must be (points, 3), or layers of them, (layers, points, 3)"
            )
        return _absolute_file(  # type: ignore[no-any-return]
            cls, array.reshape((1,) * (3 - array.ndim) + array.shape), options
        )

    @property
    def layers(self) -> tuple[ProfileLayer, ...]:
        """The layers, which the file keeps: the same objects come back on every access."""
        return self._layers

    @property
    def is_regular_grid(self) -> bool:
        """Whether the x and y axes are both incremental, so the points are evenly spaced.

        The positions are then described completely by ``x_values`` of the layers. A profile that
        stores a coordinate for each point, such as one with unevenly spaced points or one along a
        diagonal, is not regular. ``x_values`` and ``spacing`` need only an incremental x axis.

        >>> import numpy as np
        >>> import x3pio
        >>> x3pio.Profile.from_array(np.zeros(4), x_scale=1e-6).is_regular_grid
        True
        >>> x3pio.Profile.from_points(np.zeros((4, 3))).is_regular_grid
        False
        """
        return self.header.x.is_incremental and self.header.y.is_incremental


class Surface(X3pFile):
    """A surface: a matrix of points, in one or several layers.

    The data is in :attr:`layers`, one :class:`SurfaceLayer` for each layer, whose ``z`` is shaped
    ``(rows, columns)``: ``y`` increases with the row index and ``x`` with the column index.

    Make a surface on a regular grid from the heights and the steps along x and y. Then
    ``is_regular_grid`` is true, and the axes describe the grid completely::

        >>> import numpy as np
        >>> import x3pio
        >>> heights = np.zeros((480, 640))
        >>> surface = x3pio.Surface.from_array(heights, x_scale=1e-6, y_scale=1e-6)
        >>> surface.is_regular_grid, surface.layers[0].z.shape
        (True, (480, 640))
        >>> layer = surface.layers[0]
        >>> round(float(layer.x_values[-1]), 9), round(float(layer.y_values[-1]), 9)
        (0.000639, 0.000479)

    Several surfaces of the same size, as the layers of one file::

        >>> layered = x3pio.Surface.from_layers([heights, heights], x_scale=1e-6, y_scale=1e-6)
        >>> len(layered.layers), layered.layers[1].points().shape
        (2, (480, 640, 3))

    An irregular surface, whose points keep their rows and columns but not their even spacing,
    from the coordinates of its points; each layer gives them again, :meth:`~SurfaceLayer.points`::

        >>> row0 = [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]
        >>> row1 = [[0.0, 1.5, 3.0], [1.2, 1.5, 4.0]]
        >>> irregular = x3pio.Surface.from_points([row0, row1])
        >>> irregular.is_regular_grid, irregular.layers[0].points().shape
        (False, (2, 2, 3))

    A surface that lies tilted or moved in the global coordinate system keeps its stored
    coordinates and carries a placement; the layers give either system::

        >>> tilt = x3pio.Placement.from_axis_angle([1.0, 0.0, 0.0], 0.1)
        >>> tilted = surface.with_placement(tilt).layers[0]
        >>> view = tilted.points(coordinate_system=x3pio.CoordinateSystem.VIEW)
        >>> view.shape == tilted.points().shape
        True
    """

    _FEATURE: ClassVar[FeatureType] = FeatureType.SURFACE
    _NAME: ClassVar[str] = "surface"
    _LAYER: ClassVar[type[Layer]] = SurfaceLayer
    _LAYER_NDIM: ClassVar[int] = 2
    _layers: tuple[SurfaceLayer, ...]

    @classmethod
    def from_array(
        cls,
        data: ArrayLike,
        *,
        x_scale: float,
        y_scale: float | None = None,
        **options: Unpack[ArrayOptions],
    ) -> Surface:
        """Build a surface from an array of heights.

        The array is used as it is when it is already ``float64``, not copied, so do not
        change it while the file is in use.

        :param data: Heights in metres: ``(rows, columns)`` for a surface of one layer, or
            ``(layers, rows, columns)``. ``NaN`` marks a point that is not measured.
        :param x_scale: Sampling interval along x, in metres.
        :param y_scale: Sampling interval along y, in metres.
        :param options: See :class:`ArrayOptions`.
        :raises X3pFormatError: If ``data`` is not two- or three-dimensional, a scale is not
            positive, or global heights are given with a rotated placement.
        :raises TypeError: If ``y_scale`` is left out, or ``placement`` is given without
            ``coordinate_system``.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
        >>> file.layers[0].z.shape
        (2, 3)
        """
        array = np.asarray(data, dtype=np.float64)
        if array.ndim not in (2, 3):
            raise X3pFormatError(
                "Data must be a surface, (rows, columns), or layers, (layers, rows, columns)"
            )
        if y_scale is None:
            raise TypeError("y_scale is required for a surface")
        matrix = array.reshape((1,) * (3 - array.ndim) + array.shape)
        return _matrix_file(  # type: ignore[no-any-return]
            cls, matrix, matrix, x_scale=x_scale, y_scale=y_scale, options=options
        )

    @classmethod
    def from_layers(
        cls,
        layers: Iterable[ArrayLike | SurfaceLayer],
        *,
        x_scale: float | None = None,
        y_scale: float | None = None,
        **options: Unpack[ArrayOptions],
    ) -> Surface:
        """Build a surface of several layers, from arrays of heights or from layers.

        The layers have to have the same shape. Given as arrays, each is one layer of heights and
        the scales say how far apart the points are. Given as :class:`SurfaceLayer`, they bring
        their axes and their placement, which have to agree, and no scale is given.

        :param layers: Arrays of heights shaped ``(rows, columns)`` in metres, with ``NaN`` for a
            point that is not measured, or layers, for example from other files.
        :param x_scale: Sampling interval along x, in metres, for arrays.
        :param y_scale: Sampling interval along y, in metres, for arrays.
        :param options: See :class:`ArrayOptions`; with layers only ``metadata``, ``storage`` and
            ``revision`` apply.
        :raises X3pFormatError: If there is no layer, arrays are not two-dimensional or layers
            differ in their shape, their axes or their placement.
        :raises TypeError: If a scale is left out for arrays, or given with layers.

        >>> import numpy as np
        >>> import x3pio
        >>> scans = [np.zeros((3, 4)), np.ones((3, 4))]
        >>> file = x3pio.Surface.from_layers(scans, x_scale=1e-6, y_scale=1e-6)
        >>> len(file.layers), file.layers[0].z.shape
        (2, (3, 4))
        >>> float(x3pio.Surface.from_layers(list(reversed(file.layers))).layers[0].z.max())
        1.0
        """
        items = tuple(layers)
        if len(items) > 0 and all(isinstance(layer, SurfaceLayer) for layer in items):
            if x_scale is not None or y_scale is not None:
                raise TypeError("Layers bring their scales; leave x_scale and y_scale out")
            return _file_of_layers(cls, items, options)  # type: ignore[no-any-return]
        arrays = _layer_arrays(items, 2, "surface")
        if x_scale is None or y_scale is None:
            raise TypeError("x_scale and y_scale are required for arrays of heights")
        return cls.from_array(np.stack(arrays), x_scale=x_scale, y_scale=y_scale, **options)

    @classmethod
    def from_points(cls, points: ArrayLike, **options: Unpack[PointOptions]) -> Surface:
        """Build a surface from the coordinates of its points, on a grid that need not be regular.

        This is the inverse of :meth:`~Layer.points`. All three axes are absolute: every point
        stores its ``x``, ``y`` and ``z``, and the points keep the neighbours of their rows and
        columns, which the standard makes the duty of the writer: a point's neighbours in space
        have to be those in the array. Use :meth:`from_array` for a regular grid.

        :param points: ``(rows, columns, 3)`` array of ``x``, ``y``, ``z`` in metres for a surface
            of one layer, or ``(layers, rows, columns, 3)``. ``NaN`` in ``z`` marks a point
            that is not measured.
        :param options: See :class:`PointOptions`.
        :raises X3pFormatError: If ``points`` has another shape, or does not fit ``data_type``
            without overflow.
        :raises TypeError: If ``placement`` is given without ``coordinate_system``.

        >>> import numpy as np
        >>> import x3pio
        >>> row0 = [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]
        >>> row1 = [[0.0, 1.5, 3.0], [1.2, 1.5, 4.0]]
        >>> file = x3pio.Surface.from_points([row0, row1])
        >>> file.layers[0].z.shape, file.is_regular_grid
        ((2, 2), False)
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim not in (3, 4) or array.shape[-1] != 3:
            raise X3pFormatError(
                "A grid of points is (rows, columns, 3), or (layers, rows, columns, 3) for layers"
            )
        return _absolute_file(  # type: ignore[no-any-return]
            cls, array.reshape((1,) * (4 - array.ndim) + array.shape), options
        )

    @property
    def layers(self) -> tuple[SurfaceLayer, ...]:
        """The layers, which the file keeps: the same objects come back on every access."""
        return self._layers

    @property
    def is_regular_grid(self) -> bool:
        """Whether the x and y axes are both incremental, so every layer is a regular grid.

        The grid is then described completely by ``x_values`` and ``y_values`` of the layers.
        """
        return self.header.x.is_incremental and self.header.y.is_incremental


class PointCloud(X3pFile):
    """A point cloud: an unordered list of points without any topology.

    The points are in :attr:`layers`, which holds the one :class:`PointLayer` that a point cloud
    presents, with ``z``, ``x`` and ``y`` shaped ``(points,)``. All three axes are absolute and a
    point cannot be missing.

    Make a point cloud from an ``(N, 3)`` array of ``x``, ``y`` and ``z`` in metres. ``points()``
    of the layer returns an array of the same shape::

        >>> import numpy as np
        >>> import x3pio
        >>> points = np.random.default_rng(0).uniform(0.0, 1e-3, (1000, 3))
        >>> cloud = x3pio.PointCloud.from_points(points)
        >>> cloud.layer.points().shape, cloud.layer.x.shape
        ((1000, 3), (1000,))

    A point that is not measured cannot be kept, as a surface does with ``NaN``: leave it out.
    The storage type decides the precision of the file, ``float64`` by default::

        >>> compact = x3pio.PointCloud.from_points(points, data_type=x3pio.DataType.INT16)
        >>> stored = x3pio.X3pFile.loads(compact.dumps())
        >>> bool(np.abs(stored.layer.points() - points).max() < 1e-7)
        True
    """

    _FEATURE: ClassVar[FeatureType] = FeatureType.POINT_CLOUD
    _NAME: ClassVar[str] = "point cloud"
    _LAYER: ClassVar[type[Layer]] = PointLayer
    _LAYER_NDIM: ClassVar[int] = 1
    _layers: tuple[PointLayer, ...]

    @classmethod
    def _check_layers(cls, layers: Sequence[Layer]) -> None:
        super()._check_layers(layers)
        if len(layers) != 1:
            raise X3pFormatError("A point cloud has exactly one layer")

    @classmethod
    def _layers_of(
        cls,
        header: Header,
        placement: Placement,
        z: np.ndarray,
        x: np.ndarray | None,
        y: np.ndarray | None,
    ) -> tuple[Layer, ...]:
        return (PointLayer(z, x, y, header=header, placement=placement),)

    def _stacked(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        layer = self._layers[0]
        return layer.z, layer.x, layer.y

    @classmethod
    def from_points(cls, points: ArrayLike, **options: Unpack[PointOptions]) -> PointCloud:
        """Build a point cloud from an array of points.

        :param points: ``(points, 3)`` array of ``x``, ``y``, ``z`` in metres.
        :param options: See :class:`PointOptions`.
        :raises X3pFormatError: If ``points`` is not a finite ``(N, 3)`` array, or does not fit
            ``data_type`` without overflow.
        :raises TypeError: If ``placement`` is given without ``coordinate_system``.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.PointCloud.from_points(np.array([[0.0, 1.0, 2.0]]))
        >>> file.layer.points().tolist()
        [[0.0, 1.0, 2.0]]
        """
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise X3pFormatError("Points must be an (N, 3) array")
        if not np.all(np.isfinite(array)):
            raise X3pFormatError(
                "A point cloud cannot hold points that are not measured; remove them first"
            )
        return _absolute_file(cls, array, options)  # type: ignore[no-any-return]

    @property
    def layers(self) -> tuple[PointLayer, ...]:
        """The one layer that a point cloud presents."""
        return self._layers

    @property
    def layer(self) -> PointLayer:
        """The one layer of the point cloud."""
        return self.layers[0]


def parse_file(data: bytes | IO[bytes], *, verify: bool = True) -> X3pFile:
    """Parse the content of an x3p file into a profile, a surface or a point cloud.

    :param verify: Check the MD5 checksums stored in the file, of ``main.xml`` and of the binary
        files. A checksum that is missing or malformed is not checked.
    :raises X3pFormatError: If ``data`` is not an x3p file, or is damaged beyond what can be
        tolerated.
    :raises X3pChecksumError: If ``verify`` and a stored checksum does not match the content, so
        the file was changed or damaged after it was written.
    """
    from ..codec import container  # imported here: the codec imports this module

    return _from_parsed(container.parse(data, verify=verify))


def _from_parsed(fields: dict[str, Any]) -> X3pFile:
    """Make the type of file that a parsed file is, from the fields that the parser returns."""
    feature = fields.pop("feature_type")
    if feature is FeatureType.POINT_CLOUD:
        return PointCloud._from_stacked(**fields)
    if feature is FeatureType.SURFACE:
        return Surface._from_stacked(**fields)
    for name in ("z", "x", "y"):  # a profile has a single row, which the arrays leave out
        if fields[name] is not None:
            fields[name] = fields[name].reshape(fields[name].shape[0], fields[name].shape[2])
    return Profile._from_stacked(**fields)
