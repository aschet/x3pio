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

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, Self, TypedDict, TypeVar, Unpack

import numpy as np
from numpy.typing import ArrayLike

from .._version import __version__
from ..exceptions import X3pFormatError
from .arithmetic import centered, globalize, rescaled, resolve_coordinate_system
from .codec_port import codec
from .datatypes import (
    DataType,
    get_data_type,
    suggest_scale,
)
from .extensions import VendorExtensions, convert_extensions
from .geometry import CoordinateSystem, Placement
from .header import (
    DEFAULT_REVISION,
    Axis,
    AxisType,
    FeatureType,
    Header,
    Metadata,
    ProbingType,
    Revision,
    validate_increment,
)
from .layers import Layer, Point, PointLayer, ProfileLayer, SurfaceLayer
from .storage import DataStorage
from .streams import PathOrStream, read_bytes, write_bytes
from .validation import validate

__all__ = [
    "ArrayOptions",
    "MetadataChanges",
    "Point",
    "PointCloud",
    "PointOptions",
    "Profile",
    "Surface",
    "X3pFile",
]


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


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a view of ``array`` that shares its memory, but cannot be written to."""
    view = array.view()
    view.flags.writeable = False
    return view


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


class X3pFile(ABC):
    """What a profile, a surface and a point cloud share: the file-level part of an x3p file.

    There is no file of this type itself. :func:`~x3pio.read` returns a :class:`Profile`, a
    :class:`Surface` or a :class:`PointCloud`, which are made with their own ``from_`` methods.
    The data is in the :attr:`layers`; the file holds what belongs to all of them: the axes and the
    placement, which its layers share, the metadata, the vendor extensions and the storage form.

    :func:`~x3pio.read` returns the type that matches the ``FeatureType`` of the file. The
    properties and methods of this class exist on all three types, so code that handles any
    file needs no check of the type::

        >>> import io
        >>> import numpy as np
        >>> import x3pio
        >>> blob = x3pio.Surface.from_array(np.zeros((2, 3)), x_scale=1.0, y_scale=1.0).dumps()
        >>> x3p = x3pio.read(io.BytesIO(blob))
        >>> type(x3p).__name__, x3p.feature_type.name
        ('Surface', 'SURFACE')
        >>> [layer.points().shape for layer in x3p.layers]
        [(2, 3, 3)]

    ``open`` and ``loads`` of a type refuse a file of another kind::

        >>> x3pio.PointCloud.loads(blob)
        Traceback (most recent call last):
        ...
        x3pio.exceptions.X3pFormatError: The file holds a surface, not a point cloud

    The arrays of a layer belong to the file, so changing them changes the file, and it can be
    written again. The methods that return a copy, ``with_metadata``, ``with_revision``,
    ``with_z_type``, ``with_placement`` and the others, share the arrays with the file as
    read-only views; make a writable file of one with :meth:`copy`, or give a file other layers
    with :meth:`with_layers`::

        >>> layer = x3p.layers[0]
        >>> layer.z -= 1.0
        >>> described = x3p.with_metadata(comment="levelled")
        >>> float(layer.z[0, 0]), described.layers[0].z.flags.writeable
        (-1.0, False)
        >>> float(described.copy().layers[0].z[0, 0])
        -1.0

    The coordinates of a layer are in metres in the view coordinate system: the stored values
    multiplied by the axis increment. The ``placement`` says where that system lies in the
    global one, and :class:`CoordinateSystem` names the two; no function applies it implicitly.
    ``NaN`` marks a point that is not measured.

    A file keeps its layers, so ``file.layers[0]`` is the same object each time, and changing the
    arrays of a layer changes the file. The axes (:attr:`header`) and the :attr:`placement` belong
    to the layers and are immutable values; a copy with other ones is made with a ``with_`` method
    of the file. The metadata, the extensions and the storage form are attributes of the file that
    can be changed.

    The constructor is the low-level one and takes the layers as they are, without copying them:
    :meth:`from_stacked` makes a file from stacked arrays, and the ``from_`` methods of the types
    from heights or points.

    :param layers: The layers of the file, of the layer class of its type: they have to agree in
        shape, axes and placement, and a point cloud has one.
    :param revision: The revision of the standard that the file follows, see :class:`Revision`;
        Amd 1:2020 by default. Convert a file to another with :meth:`with_revision`.
    :param metadata: ``Record2``, or ``None`` if the file has none.
    :param extensions: The vendor specific extension files, by their ``VendorSpecificID`` such
        as ``"http://www.vendor.com/mypath/a.xml"``: a mapping to bytes, with ``None`` for an ID
        without a file, see :class:`VendorExtensions`. They are kept when a file is read and
        written again, and not interpreted. Without them the file has none.
    :param storage: Whether the coordinates are written as XML text or binary.
        Not part of the comparison of two files.
    :raises X3pFormatError: If the layers do not agree, or there are none.
    :raises TypeError: If a layer is not of the layer class of the type.
    """

    #: The class of the data, as written in the ``FeatureType`` element; set by each type.
    _FEATURE: ClassVar[FeatureType]
    #: The name of the type in messages.
    _NAME: ClassVar[str]
    #: The class of the layers of each type.
    _LAYER: ClassVar[type[Layer]]
    #: The number of dimensions of ``z`` of a layer of each type.
    _LAYER_NDIM: ClassVar[int]

    def __init__(
        self,
        layers: Iterable[Layer],
        *,
        revision: Revision = DEFAULT_REVISION,
        metadata: Metadata | None = None,
        extensions: VendorExtensions | None = None,
        storage: DataStorage = DataStorage.BINARY,
    ) -> None:
        """Keep the layers, and give empty extensions the revision of the file."""
        stored = tuple(layers)
        self._check_layers(stored)
        self._layers = stored
        self._revision = revision
        self.metadata = metadata
        if extensions is None or (
            extensions._revision is not revision and extensions._parts() == ([], {})
        ):
            extensions = VendorExtensions(revision)
        self.extensions = extensions
        self.storage = storage

    @classmethod
    def _check_layers(cls, layers: Sequence[Layer]) -> None:
        """Refuse layers that are none, of another kind, or do not fit together."""
        name = cls._NAME
        if len(layers) == 0:
            raise X3pFormatError(f"A {name} needs at least one layer")
        if any(type(layer) is not cls._LAYER for layer in layers):
            raise TypeError(f"The layers of a {name} must be {cls._LAYER.__name__} objects")
        first = layers[0]
        if any(
            not (layer.header is first.header or layer.header == first.header)
            or not (layer.placement is first.placement or layer.placement == first.placement)
            for layer in layers
        ):
            raise X3pFormatError(f"The layers of a {name} must have the same axes and placement")
        if any(layer.z.shape != first.z.shape for layer in layers):
            raise X3pFormatError(f"The layers of a {name} must have the same shape")
        if any(
            (layer.x is None) != (first.x is None) or (layer.y is None) != (first.y is None)
            for layer in layers
        ):
            raise X3pFormatError(f"The layers of a {name} must all have absolute axes, or none")

    @classmethod
    def _layers_of(
        cls,
        header: Header,
        placement: Placement,
        z: np.ndarray,
        x: np.ndarray | None,
        y: np.ndarray | None,
    ) -> tuple[Layer, ...]:
        """Make the layers of stacked arrays, which they view."""
        arrays = [np.asarray(array) for array in (z, x, y) if array is not None]
        if any(array.ndim == 0 for array in arrays):
            raise X3pFormatError(f"The data of a {cls._NAME} must be arrays, not single numbers")
        if any(len(array) != len(arrays[0]) for array in arrays):
            raise X3pFormatError("z, x and y need the same number of layers")
        z = np.asarray(z)
        return tuple(
            cls._LAYER(
                z[index, ...],
                None if x is None else np.asarray(x)[index, ...],
                None if y is None else np.asarray(y)[index, ...],
                header=header,
                placement=placement,
            )
            for index in range(len(z))
        )

    @classmethod
    def from_stacked(
        cls,
        header: Header,
        z: np.ndarray,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        placement: Placement | None = None,
        revision: Revision = DEFAULT_REVISION,
        metadata: Metadata | None = None,
        extensions: VendorExtensions | None = None,
        storage: DataStorage = DataStorage.BINARY,
    ) -> Self:
        """Make a file of the arrays of all its layers stacked, as a file stores them.

        This is the low-level way to make a file, for code that holds the arrays as the format
        has them. The layers view the arrays, which are not copied. The ``from_array`` and
        ``from_points`` methods of the types are made for everything else.

        :param header: The axes of the file.
        :param z: The heights of all layers stacked: ``(layers, points)`` for a profile,
            ``(layers, rows, columns)`` for a surface and ``(points,)`` for a point cloud.
        :param x: X coordinates shaped like ``z`` if the x axis is absolute, otherwise ``None``
            (the coordinate is derived from the place of a point).
        :param y: Y coordinates, like ``x``.
        :param placement: Where the view coordinate system lies in the global one, see
            :class:`Placement`; the identity by default.
        :param revision: See the constructor.
        :param metadata: See the constructor.
        :param extensions: See the constructor.
        :param storage: See the constructor.
        :raises X3pFormatError: If ``z``, ``x`` and ``y`` do not have the same number of layers.

        >>> import numpy as np
        >>> import x3pio
        >>> header = x3pio.Header(x=x3pio.Axis(increment=1e-6), y=x3pio.Axis(increment=1e-6))
        >>> surface = x3pio.Surface.from_stacked(header, np.zeros((2, 3, 4)))
        >>> len(surface.layers), surface.layers[1].z.shape
        (2, (3, 4))
        """
        layers = cls._layers_of(
            header, Placement.identity() if placement is None else placement, z, x, y
        )
        return cls(
            layers, revision=revision, metadata=metadata, extensions=extensions, storage=storage
        )

    @property
    def header(self) -> Header:
        """The axes of the file, shared by the layers; an immutable value."""
        return self._layers[0].header

    @property
    def revision(self) -> Revision:
        """The revision of the standard that the file follows, see :meth:`with_revision`."""
        return self._revision

    @property
    def placement(self) -> Placement:
        """Where the view coordinate system lies in the global one; shared by the layers."""
        return self._layers[0].placement

    @property
    @abstractmethod
    def layers(self) -> Sequence[Layer]:
        """The layers, which hold the data: several for a profile or a surface, one for a cloud.

        Each layer gives its ``z``, ``x``, ``y``, its ``valid`` mask and its ``points``. The file
        keeps them: the same objects come back on every access.
        """

    @property
    def feature_type(self) -> FeatureType:
        """The class of the data, as written in the ``FeatureType`` element."""
        return self._FEATURE

    def __eq__(self, other: object) -> bool:
        """Compare type, layers (NaN-aware), metadata and extensions for equality."""
        if not isinstance(other, X3pFile):
            return NotImplemented
        return (
            type(self) is type(other)
            and self._layers == other._layers
            and self.metadata == other.metadata
            and self.extensions == other.extensions
        )

    def __repr__(self) -> str:
        """Show the type, the layers and what the file holds besides them."""
        return (
            f"{type(self).__name__}(layers={self._layers!r}, metadata={self.metadata!r}, "
            f"extensions={self.extensions!r}, storage={self.storage!r})"
        )

    def _copy(
        self,
        *,
        header: Header | None = None,
        placement: Placement | None = None,
        revision: Revision | None = None,
        metadata: Metadata | None = None,
        extensions: VendorExtensions | None = None,
        arrays: tuple[np.ndarray, np.ndarray | None, np.ndarray | None] | None = None,
    ) -> Self:
        """Return a copy that shares nothing mutable with this file but its arrays.

        The metadata and the extensions are copied, so changing one does not change the other.
        The arrays are shared as read-only views, which saves memory for a copy that only changes
        a field; writing to them raises an error. The arguments replace what the copy gets:
        ``arrays`` are ``z``, ``x`` and ``y`` stacked as the type holds them, for data that has
        been computed anew.
        """
        header = self.header if header is None else header
        placement = self.placement if placement is None else placement
        layers: tuple[Layer, ...]
        if arrays is None:
            layers = tuple(
                layer._derived(header=header, placement=placement, arrays="read_only")
                for layer in self._layers
            )
        else:
            layers = self._layers_of(header, placement, *arrays)
        return self._rebuilt(layers, revision=revision, metadata=metadata, extensions=extensions)

    def _rebuilt(
        self,
        layers: Sequence[Layer],
        *,
        revision: Revision | None = None,
        metadata: Metadata | None = None,
        extensions: VendorExtensions | None = None,
    ) -> Self:
        """Return a file of ``layers`` with a copy of what this file holds besides them."""
        return type(self)(
            layers,
            revision=self._revision if revision is None else revision,
            metadata=deepcopy(self.metadata) if metadata is None else metadata,
            extensions=self.extensions._copy() if extensions is None else extensions,
            storage=self.storage,
        )

    def _stacked(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``z``, ``x`` and ``y`` of all layers stacked, as ``from_stacked`` takes them.

        A single layer is stacked as a view, several are copied.
        """

        def stack(arrays: list[np.ndarray]) -> np.ndarray | None:
            if not arrays:
                return None
            return arrays[0][None] if len(arrays) == 1 else np.stack(arrays)

        layers = self._layers
        z = stack([layer.z for layer in layers])
        x = stack([layer.x for layer in layers if layer.x is not None])
        y = stack([layer.y for layer in layers if layer.y is not None])
        return z, x, y  # type: ignore[return-value]

    def _cube(self) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``z``, ``x`` and ``y`` shaped ``(layers, rows, columns)`` as the format has it.

        A point cloud is a list and stays one-dimensional. The arrays are views if there is a
        single layer.
        """
        return self._stacked()

    def _uncube(
        self, data: np.ndarray, x: np.ndarray | None, y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return arrays shaped as the format has them in the shape of this type, see ``_cube``."""
        return data, x, y

    def with_z_type(
        self, data_type: DataType, *, increment: float | Literal["auto"] = "auto"
    ) -> Self:
        """Return a copy that stores the heights in a different type.

        The copy is independent, except that it shares the arrays with this file as
        read-only views.

        :param data_type: The new storage type of the z axis.
        :param increment: The new z increment, or ``"auto"`` for the smallest
            that fits the heights into ``data_type`` without overflow
            (``1`` for a floating point type).
        :raises X3pFormatError: If the increment is not positive.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.array([[0.0, 1e-6]]), x_scale=1e-6, y_scale=1e-6)
        >>> file.with_z_type(x3pio.DataType.INT16).header.z.data_type.name
        'INT16'
        """
        resolved = get_data_type(data_type)
        resolved_increment = (
            suggest_scale(self._cube()[0], resolved) if increment == "auto" else increment
        )
        validate_increment(resolved_increment)
        z = replace(self.header.z, data_type=resolved.type, increment=resolved_increment)
        return self._copy(header=replace(self.header, z=z))

    def globalized(self) -> Self:
        """Return a copy whose stored coordinates are the global ones.

        The placement is applied to the coordinates once, so that ``z``, ``x`` and ``y`` of
        the result are the global coordinates and its placement is the identity, except for
        incremental axes: their offset is the position of the first column or row and stays in
        the placement (the ``x_values`` of a layer plus ``placement.offset[0]`` are the global
        x coordinates).

        Without rotation, the result keeps the matrix and its
        regular grid. An axis with a stored integer type and an offset
        becomes ``float64``, since the shifted values no longer fall on
        multiples of its increment. With a rotation, a regular grid is turned
        into a skew one, so the result has absolute axes with ``x``, ``y`` and
        ``z`` for all points.

        The points of a layer of the result in ``CoordinateSystem.VIEW`` equal those of the same
        layer of this file in the global coordinate system. The result is independent of this file,
        except that it may share arrays with it as read-only views.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
        >>> moved = file.with_placement(x3pio.Placement.translation(0.0, 0.0, 5.0))
        >>> moved.globalized().layers[0].z.tolist()
        [[5.0, 5.0]]
        """
        header, placement, z, x, y = globalize(self.header, self.placement, *self._cube())
        return self._copy(header=header, placement=placement, arrays=self._uncube(z, x, y))

    def with_placement(self, placement: Placement) -> Self:
        """Return a copy with another placement; the stored coordinates stay as they are.

        The points move in the global coordinate system, since the view system that they are
        stored in is now somewhere else. Use :meth:`transformed` to move the points
        relative to where they are now.

        :param placement: Where the view coordinate system lies in the global one.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Profile.from_array(np.zeros(2), x_scale=1.0)
        >>> moved = file.with_placement(x3pio.Placement.translation(5.0, 0.0, 0.0))
        >>> moved.layers[0].points()[:, 0].tolist()
        [5.0, 6.0]
        """
        return self._copy(placement=placement)

    def transformed(self, transform: Placement) -> Self:
        """Return a copy whose points are moved in the global coordinate system.

        The global coordinates become ``transform.apply(old global coordinates)``, and the
        stored coordinates stay as they are: the new placement is ``transform @ placement``.

        :param transform: The rigid transformation to apply to the points.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Profile.from_array(np.zeros(2), x_scale=1.0)
        >>> up = x3pio.Placement.translation(0.0, 0.0, 1.0)
        >>> file.transformed(up).layers[0].points()[:, 2].tolist()
        [1.0, 1.0]
        """
        return self._copy(placement=transform @ self.placement)

    def centered(self) -> Self:
        """Return a copy whose stored coordinates are centred on the origin.

        The standard recommends choosing the offset so that the stored points are centred on the
        origin of the coordinate system. Each absolute axis is shifted by the middle of its
        range, and the placement moves by the same amount, so the global coordinates stay as
        they are. An incremental axis counts from the first point and is left alone.

        The increment of an axis that is stored as an integer is chosen again for the centred
        range, as ``"auto"`` does, since the symmetric range uses the type best and the increment
        for the old range would waste most of it. An increment that was given for such an axis is
        replaced. The increments of other axes are not changed.

        >>> import numpy as np
        >>> import x3pio
        >>> cloud = x3pio.PointCloud.from_points(np.array([[10.0, 0.0, 5.0], [12.0, 2.0, 7.0]]))
        >>> centered = cloud.centered()
        >>> centered.layer.x.tolist(), centered.placement.offset.tolist()
        ([-1.0, 1.0], [11.0, 1.0, 6.0])
        >>> centered.layer.points().tolist() == cloud.layer.points().tolist()
        True
        """
        placement, z, x, y = centered(self.header, self.placement, *self._cube())
        header = rescaled(self.header, z, x, y)
        return self._copy(header=header, placement=placement, arrays=self._uncube(z, x, y))

    def copy(self) -> Self:
        """Return a copy that shares nothing with this file, with arrays that can be written to.

        The ``with_`` methods share the arrays of the file with their result as read-only views,
        to save memory. Make a copy of such a result to change its values in place.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
        >>> edited = file.with_metadata(comment="edited").copy()
        >>> edited.layers[0].z[0, 0] = 1.0
        >>> float(file.layers[0].z[0, 0]), float(edited.layers[0].z[0, 0])
        (0.0, 1.0)
        """
        return self._rebuilt(tuple(layer._derived(arrays="copy") for layer in self._layers))

    def with_layers(self, layers: Iterable[Layer]) -> Self:
        """Return a file of other layers that keeps everything else of this one.

        The metadata, the vendor extensions, the revision and the storage form stay; the layers
        bring their axes, their placement and their values, and are copied. This is the way to
        write back data whose shape or points differ from the ones that were read, such as a
        cropped array or a surface of other points, without losing the metadata.

        :param layers: The layers of the new file, of the layer class of its type: they have to
            agree in shape, axes and placement, and a point cloud has one.
        :raises X3pFormatError: If there is no layer, or the layers do not agree.
        :raises TypeError: If a layer is not of the layer class of the type.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.ones((4, 4)), x_scale=1e-6, y_scale=1e-6)
        >>> cropped = x3pio.SurfaceLayer.from_array(
        ...     file.layers[0].z[:2, :2], x_scale=1e-6, y_scale=1e-6
        ... )
        >>> smaller = file.with_layers([cropped])
        >>> smaller.layers[0].z.shape, smaller.metadata == file.metadata
        ((2, 2), True)
        """
        stored = tuple(layers)
        self._check_layers(stored)
        return self._rebuilt([layer._derived(arrays="copy") for layer in stored])

    def with_metadata(self, **changes: Unpack[MetadataChanges]) -> Self:
        """Return a copy with changed metadata (``Record2``); this file stays as it is.

        The copy is independent, except that it shares the arrays with this file as
        read-only views. A file without metadata is given a new record first, with the current UTC
        time as date, an empty instrument, and the probing type ``ProbingType.SOFTWARE``; set the
        fields that describe the data set truthfully. To replace the whole record, assign
        ``file.metadata``.

        :param changes: The fields to change, see :class:`MetadataChanges`.
        :returns: The changed copy.
        :raises TypeError: If a field is not one of :class:`MetadataChanges`.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1e-6, y_scale=1e-6)
        >>> described = file.with_metadata(creator="Jane Doe", comment="Test surface")
        >>> described.metadata.creator, file.metadata.creator
        ('Jane Doe', None)
        """
        return self._copy(metadata=_apply_metadata_changes(self.metadata, changes))

    def with_revision(
        self,
        revision: Revision,
        *,
        drop_extensions: bool = False,
    ) -> Self:
        """Return a copy that follows another revision of the standard.

        Data and metadata are kept, but ``ISO5436_2000`` requires a calibration date, which is
        taken from the date of creation if there is none. The arrays are shared as read-only views.
        The vendor extensions are converted to Amd 1:2020, where each file becomes an ID of the
        vendor, a ``/`` and its path. They are not converted to ``ISO5436_2000``; see
        ``drop_extensions``.

        :param revision: The revision to convert to.
        :param drop_extensions: Drop the extensions when converting to ``ISO5436_2000``, instead
            of failing.
        :raises X3pFormatError: If there are extensions that would have to be dropped, and
            ``drop_extensions`` is not set, or a file cannot get an ID of its own.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
        >>> file.revision
        <Revision.ISO25178_72_2017_DAM1: 'ISO25178-72:2017/DAM1'>
        >>> file.with_revision(x3pio.Revision.ISO5436_2000).revision.value
        'ISO5436 - 2000'
        """
        if revision is self.revision:
            return self._copy()
        extensions = convert_extensions(self.extensions, revision, drop=drop_extensions)
        metadata = deepcopy(self.metadata)
        if (
            revision.requires_calibration_date
            and metadata is not None
            and metadata.calibration_date is None
        ):
            metadata.calibration_date = metadata.date  # the schema of the revision requires one
        return self._copy(revision=revision, extensions=extensions, metadata=metadata)

    @classmethod
    def loads(cls, data: bytes, *, verify: bool = True) -> Self:
        """Parse an in-memory x3p file.

        A set of known deviations from the standard is tolerated, see :mod:`x3pio.codec`;
        other deviations raise an error.

        :param data: The content of an ``.x3p`` file.
        :param verify: Check the MD5 checksums stored in the file, of ``main.xml`` and of the
            binary files. A checksum that is missing or malformed is not checked.
        :returns: A :class:`Profile`, a :class:`Surface` or a :class:`PointCloud`, as the file
            says. Called on one of them, the file has to be of that type.
        :raises X3pFormatError: If ``data`` is not an x3p file, or is damaged
            beyond what can be tolerated, or is not the type this was called on.
        :raises X3pChecksumError: If ``verify`` and a stored checksum does not match the
            content, so the file was changed or damaged after it was written.
        """
        file = _from_parsed(codec().parse(data, verify=verify))
        if not isinstance(file, cls):
            raise X3pFormatError(
                f"The file holds a {file.feature_type.name.lower().replace('_', ' ')}, "
                f"not a {'point cloud' if cls is PointCloud else cls.__name__.lower()}"
            )
        return file

    @classmethod
    def open(cls, path: PathOrStream, *, verify: bool = True) -> Self:
        """Read an x3p file.

        :param path: Path to an ``.x3p`` file, or an open binary file object.
        :param verify: See :meth:`loads`.
        :raises X3pFormatError: See :meth:`loads`.
        :raises X3pChecksumError: See :meth:`loads`.
        """
        return cls.loads(read_bytes(path), verify=verify)

    def _validate(self) -> None:
        """Check that the file can be written as a file that conforms to the standard."""
        validate(self)

    def _records(self) -> np.ndarray:
        return codec().records(self)

    def raw_data(self) -> np.ndarray:
        """Coded coordinates in the file's native on-disk layout.

        This is what a binary data file holds: one record per point in storage
        order (``x`` fastest), with a field ``"x"``, ``"y"`` and ``"z"`` for
        each absolute axis, in the little-endian type of the axis. Each is
        ``coordinate / increment``, rounded for an integer type. A point that
        is not measured is ``NaN`` in a floating point field, and ``0`` in an
        integer one, which relies on the validity file instead.
        Recomputed on each call, so it always reflects in-place edits.

        :raises X3pFormatError: If the file cannot be written (see :meth:`dumps`),
            for example if a value does not fit the type of its axis.
        """
        self._validate()
        return self._records()

    def dumps(self, *, storage: DataStorage | None = None) -> bytes:
        """Serialize this file to an in-memory blob, overriding the storage form if given.

        :param storage: Defaults to the form the file was read with (or
            ``storage`` if constructed directly).
        :returns: The serialized ``.x3p`` file contents. They always conform to
            the standard, whatever deviations the file had when read.
        :raises X3pFormatError: If the file cannot be written as a conforming
            file: inconsistent data or axes, a value that does
            not fit its storage type, or missing required metadata.
        :raises TypeError: If ``storage`` is given and is not a
            :class:`DataStorage` member.
        """
        if storage is not None and not isinstance(storage, DataStorage):
            raise TypeError(f"storage must be a DataStorage member, got {storage!r}")
        return codec().serialize(self, storage)

    def save(self, path: PathOrStream, *, storage: DataStorage | None = None) -> None:
        """Write this file to disk or a binary stream, optionally overriding the storage form.

        :param path: Destination path, or an open binary file object.
        :param storage: Defaults to the form the file was read with (or
            ``storage`` if constructed directly).
        :raises X3pFormatError: See :meth:`dumps`.
        """
        write_bytes(path, self.dumps(storage=storage))


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
    increment = suggest_scale(matrix, resolved_type) if z_scale == "auto" else z_scale
    header = Header(
        x=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, x_scale),
        y=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, y_scale),
        z=Axis(AxisType.ABSOLUTE, resolved_type.type, increment),
    )
    metadata = options.get("metadata")
    file = cls.from_stacked(
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
            suggest_scale(points[..., index], resolved_type) if scale == "auto" else scale,
        )
        for index in range(3)
    ]
    header = Header(x=axes[0], y=axes[1], z=axes[2])
    metadata = options.get("metadata")
    file = cls.from_stacked(
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
        >>> stored = x3pio.PointCloud.loads(compact.dumps())
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


def _from_parsed(fields: dict[str, Any]) -> X3pFile:
    """Make the type of file that a parsed file is, from the fields that the parser returns."""
    feature = fields.pop("feature_type")
    if feature is FeatureType.POINT_CLOUD:
        return PointCloud.from_stacked(**fields)
    if feature is FeatureType.SURFACE:
        return Surface.from_stacked(**fields)
    for name in ("z", "x", "y"):  # a profile has a single row, which the arrays leave out
        if fields[name] is not None:
            fields[name] = fields[name].reshape(fields[name].shape[0], fields[name].shape[2])
    return Profile.from_stacked(**fields)
