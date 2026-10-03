# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""x3pio: read and write ISO 25178-72 x3p files.

x3p is the format of ISO 25178-72:2017 and Amd 1:2020 for the storage and exchange of topography
and profile data. It stores profiles, surfaces on a regular or an irregular grid, in one or several
layers, and point clouds as 3-D coordinates in a right-handed coordinate system, together with the
metadata of the measurement and optional files of a vendor.

Basic usage::

    >>> import io
    >>> import numpy as np
    >>> import x3pio
    >>> buf = io.BytesIO()
    >>> x3pio.write(buf, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
    >>> x3p = x3pio.read(io.BytesIO(buf.getvalue()))
    >>> type(x3p).__name__, len(x3p.layers)
    ('Surface', 1)
    >>> x3p.layers[0].z.shape
    (2, 3)
    >>> x3p.header.x.increment
    1e-06
"""

from __future__ import annotations

from ._version import __version__
from .application import info, read, write, write_points
from .codec import container as _container
from .exceptions import X3pChecksumError, X3pError, X3pFormatError
from .model.codec_port import register as _register_codec
from .model.datatypes import (
    DATA_TYPES,
    DataType,
    DataTypeInfo,
    decode_raw,
    encode_raw,
    get_data_type,
    suggest_scale,
)
from .model.extensions import VendorExtensions
from .model.files import (
    ArrayOptions,
    MetadataChanges,
    Point,
    PointCloud,
    PointOptions,
    Profile,
    Surface,
    X3pFile,
)
from .model.geometry import CoordinateSystem, Placement, validate_rotation
from .model.header import (
    Axis,
    AxisType,
    FeatureType,
    Header,
    Instrument,
    Metadata,
    ProbingSystem,
    ProbingType,
    Revision,
    validate_increment,
)
from .model.info import FileDescription, FileInfo
from .model.layers import (
    Layer,
    PointLayer,
    ProfileLayer,
    SurfaceLayer,
)
from .model.storage import DataStorage

# The file types read and write through the codec that is registered here.
_register_codec(_container)

__all__ = [
    "DATA_TYPES",
    "ArrayOptions",
    "Axis",
    "AxisType",
    "CoordinateSystem",
    "DataStorage",
    "DataType",
    "DataTypeInfo",
    "FeatureType",
    "FileDescription",
    "FileInfo",
    "Header",
    "Instrument",
    "Layer",
    "Metadata",
    "MetadataChanges",
    "Placement",
    "Point",
    "PointCloud",
    "PointLayer",
    "PointOptions",
    "ProbingSystem",
    "ProbingType",
    "Profile",
    "ProfileLayer",
    "Revision",
    "Surface",
    "SurfaceLayer",
    "VendorExtensions",
    "X3pChecksumError",
    "X3pError",
    "X3pFile",
    "X3pFormatError",
    "__version__",
    "decode_raw",
    "encode_raw",
    "get_data_type",
    "info",
    "read",
    "suggest_scale",
    "validate_increment",
    "validate_rotation",
    "write",
    "write_points",
]
