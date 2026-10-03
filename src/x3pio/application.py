# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Functions to read, inspect and write x3p files.

The types of the data are in ``x3pio.model``.
"""

from __future__ import annotations

from typing import Unpack

import numpy as np
from numpy.typing import ArrayLike

from .codec import container
from .exceptions import X3pFormatError
from .model.files import (
    ArrayOptions,
    PointCloud,
    PointOptions,
    Profile,
    Surface,
    X3pFile,
)
from .model.info import FileInfo
from .model.streams import PathOrStream

__all__ = ["info", "read", "write", "write_points"]


def read(path: PathOrStream, *, verify: bool = True) -> X3pFile:
    """Read an x3p file into memory.

    A set of known deviations from the standard is tolerated, see :mod:`x3pio.codec`; other
    deviations raise an error.

    :param path: Path to an ``.x3p`` file, or an open binary file object.
    :param verify: Check the MD5 checksums stored in the file, see :meth:`X3pFile.loads`.
    :returns: A :class:`Profile`, a :class:`Surface` or a :class:`PointCloud`, with the
        coordinates in metres (``NaN`` for points that are not measured), metadata and
        extension files.
    :raises X3pFormatError: If the file is not an x3p file, or is damaged
        beyond what can be tolerated.
    :raises X3pChecksumError: If ``verify`` and a stored checksum does not match.
    """
    return X3pFile.open(path, verify=verify)


def info(path: PathOrStream, *, verify: bool = True) -> FileInfo:
    """Read the description of an x3p file without its coordinates.

    The zip directory, ``main.xml`` and the vendor specific files are read. The binary data and
    validity files are not, so damage to them is not detected.

    :param path: Path to an ``.x3p`` file, or an open binary file object; a stream that can seek
        is read only where needed.
    :param verify: Check the MD5 checksum of the description, ``main.xml``. The checksums of the
        data files are not checked, since the data is not read.
    :returns: The description as a :class:`FileInfo`.
    :raises X3pFormatError: If the file is not an x3p file, or is damaged
        beyond what can be tolerated.
    :raises X3pChecksumError: If ``verify`` and the checksum of ``main.xml`` does not match.

    >>> import io
    >>> import numpy as np
    >>> import x3pio
    >>> buf = io.BytesIO()
    >>> x3pio.write(buf, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
    >>> summary = x3pio.info(buf)
    >>> summary.feature_type.name, summary.shape, summary.layer_count
    ('SURFACE', (2, 3), 1)
    """
    return container.describe(path, verify=verify)


def write(
    path: PathOrStream,
    data: ArrayLike,
    *,
    x_scale: float,
    y_scale: float | None = None,
    **options: Unpack[ArrayOptions],
) -> None:
    """Write a NumPy array of heights to an x3p file.

    :param path: Destination path, or an open binary file object.
    :param data: Heights in metres: ``(points,)`` for a profile, ``(rows, columns)`` for
        a surface or ``(layers, rows, columns)`` for a surface of several layers. ``NaN`` marks
        a point that is not measured. To write a profile of several layers, make it with
        :meth:`Profile.from_array` and save it.
    :param x_scale: Sampling interval along x, in metres.
    :param y_scale: Sampling interval along y, in metres; required for a surface, and not given
        for a profile.
    :param options: See :class:`ArrayOptions`. To leave the metadata out, build the file with
        :meth:`Surface.from_array`, set its ``metadata`` to ``None`` and save it.
    :raises X3pFormatError: If ``data`` has the wrong shape, a scale is not
        positive, ``data`` does not fit ``z_type`` without overflow, or the offset or
        rotation is invalid.
    :raises TypeError: If ``placement`` is given without ``coordinate_system``.

    ``path`` also accepts an open binary stream, e.g. to avoid touching disk::

        >>> import io
        >>> import numpy as np
        >>> import x3pio
        >>> buf = io.BytesIO()
        >>> x3pio.write(buf, np.zeros((2, 3)), x_scale=1e-6, y_scale=1e-6)
        >>> file = x3pio.read(io.BytesIO(buf.getvalue()))
        >>> file.layers[0].z.shape
        (2, 3)
    """
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        if y_scale is not None:
            raise TypeError("A profile has no y_scale; leave it out")
        Profile.from_array(array, x_scale=x_scale, **options).save(path)
    elif array.ndim in (2, 3):
        Surface.from_array(array, x_scale=x_scale, y_scale=y_scale, **options).save(path)
    else:
        raise X3pFormatError("Data must be a 1-D profile, a 2-D surface or a 3-D array of layers")


def write_points(path: PathOrStream, points: ArrayLike, **options: Unpack[PointOptions]) -> None:
    """Write an array of 3-D points to an x3p point cloud file.

    :param path: Destination path, or an open binary file object.
    :param points: ``(points, 3)`` array of ``x``, ``y``, ``z`` in metres.
    :param options: See :class:`PointOptions`.
    :raises X3pFormatError: If ``points`` is not a finite ``(N, 3)`` array, or does not fit
        ``data_type`` without overflow.
    :raises TypeError: If ``placement`` is given without ``coordinate_system``.

    >>> import io
    >>> import numpy as np
    >>> import x3pio
    >>> buf = io.BytesIO()
    >>> x3pio.write_points(buf, np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]))
    >>> x3pio.read(io.BytesIO(buf.getvalue())).layers[0].points().tolist()
    [[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]
    """
    PointCloud.from_points(points, **options).save(path)
