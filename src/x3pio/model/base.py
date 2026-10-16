# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The base of the file types: what a profile, a surface and a point cloud share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any, ClassVar, Literal, Self, Unpack

import numpy as np
from numpy.typing import NDArray

from ..exceptions import X3pFormatError
from .arithmetic import centered, globalize, rescaled
from .datatypes import (
    DataType,
    get_data_type,
    suggest_scale,
)
from .extensions import VendorExtensions, convert_extensions
from .geometry import Placement
from .header import (
    DEFAULT_REVISION,
    FeatureType,
    Header,
    Metadata,
    Revision,
    validate_increment,
)
from .layers import Layer
from .options import MetadataChanges, _apply_metadata_changes
from .storage import DataStorage
from .streams import PathOrStream, write_bytes
from .validation import validate

__all__ = ["X3pFile"]


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
            suggest_scale(self._cube()[0], resolved.type) if increment == "auto" else increment
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

    def extensions_fit(self, revision: Revision) -> bool:
        """Return whether the vendor extensions are kept by :meth:`with_revision`.

        That is the case without extensions, in the revision the file already follows, and in
        ``revision`` if the extensions can be converted to it. They cannot be converted to
        ``ISO5436_2000``, where ``with_revision`` fails for them unless it drops them.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
        >>> file.extensions_fit(x3pio.Revision.ISO5436_2000)
        True
        >>> file.extensions["http://vendor.example/a.xml"] = b"<a/>"
        >>> file.extensions_fit(x3pio.Revision.ISO5436_2000)
        False
        """
        if revision is self.revision:
            return True
        try:
            convert_extensions(self.extensions, revision)
        except X3pFormatError:
            return False
        return True

    @staticmethod
    def loads(data: bytes, *, verify: bool = True) -> X3pFile:
        """Parse an in-memory x3p file; the counterpart of :meth:`dumps`.

        :func:`~x3pio.read` reads a path or a stream.

        :param data: The content of an ``.x3p`` file.
        :param verify: Check the MD5 checksums stored in the file, of ``main.xml`` and of the
            binary files. A checksum that is missing or malformed is not checked.
        :returns: A :class:`Profile`, a :class:`Surface` or a :class:`PointCloud`, as the file
            says.
        :raises X3pFormatError: If ``data`` is not an x3p file, or is damaged beyond what can be
            tolerated.
        :raises X3pChecksumError: If ``verify`` and a stored checksum does not match the
            content, so the file was changed or damaged after it was written.
        """
        from .files import parse_file  # imported here: the file types are made from this class

        return parse_file(data, verify=verify)

    def _validate(self) -> None:
        """Check that the file can be written as a file that conforms to the standard."""
        validate(self)

    def _records(self) -> np.ndarray:
        from ..codec import container  # imported here: the codec imports this module

        return container.records(self)

    def raw_data(self) -> NDArray[Any]:
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
        from ..codec import container  # imported here: the codec imports this module

        return container.serialize(self, storage)

    def save(self, path: PathOrStream, *, storage: DataStorage | None = None) -> None:
        """Write this file to disk or a binary stream, optionally overriding the storage form.

        :param path: Destination path, or an open binary file object.
        :param storage: Defaults to the form the file was read with (or
            ``storage`` if constructed directly).
        :raises X3pFormatError: See :meth:`dumps`.
        """
        write_bytes(path, self.dumps(storage=storage))
