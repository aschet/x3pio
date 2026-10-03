# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Conditions for writing a file that conforms to the standard.

Reading is tolerant: it takes files that other software wrote and that deviate from the standard.
Writing is not, so everything that is checked here has to hold for a file that is saved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import X3pFormatError
from .geometry import validate_rotation
from .header import FeatureType, validate_increment

if TYPE_CHECKING:
    from .files import X3pFile

__all__ = ["validate"]


def validate(file: X3pFile) -> None:
    """Check that the file can be written as a file that conforms to the standard."""
    header = file.header
    feature = file.feature_type
    ndim = file._layers[0].z.ndim
    if ndim != file._LAYER_NDIM:
        raise X3pFormatError(
            f"The heights of each layer of a {file._NAME} must be "
            f"{file._LAYER_NDIM}-D, got {ndim}-D"
        )
    data, x, y = file._cube()
    shape = data.shape
    if header.z.is_incremental:
        raise X3pFormatError("The z axis must be absolute")
    for name, axis, stored in (("x", header.x, x), ("y", header.y, y)):
        if axis.is_incremental and stored is not None:
            raise X3pFormatError(f"The {name} axis is incremental, but has coordinates")
        if not axis.is_incremental and (stored is None or stored.shape != shape):
            raise X3pFormatError(f"The absolute {name} axis needs coordinates shaped like data")
    for axis in (header.x, header.y, header.z):
        validate_increment(axis.increment)
    if not np.all(np.isfinite(file.placement.offset)):
        raise X3pFormatError("An axis offset must be a finite number")
    validate_rotation(file.placement.rotation)
    if feature is FeatureType.POINT_CLOUD:
        if any(axis.is_incremental for axis in (header.x, header.y)):
            raise X3pFormatError("The axes of a point cloud must be absolute")
        if not all(
            stored is not None and np.all(np.isfinite(stored)) for stored in (x, y)
        ) or not np.all(np.isfinite(data)):
            raise X3pFormatError("A point cloud cannot hold points that are not measured")
    if file.extensions._revision is not file.revision:
        raise X3pFormatError(
            f"The vendor extensions are those of the revision {file.extensions._revision.name}, "
            f"but the file is in the revision {file.revision.name}; "
            "convert the file with with_revision()"
        )
    file.extensions._parts()  # an extension must not take the name of a file of the format
