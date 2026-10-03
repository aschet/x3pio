# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Storage data types of the axes and their conversion to metres.

A file stores the numbers of an absolute axis as one of four types: a 16-bit or a 32-bit
integer, or a 32-bit or a 64-bit floating point number, always with the least significant byte
first. The stored number times the increment of the axis is the coordinate in metres, so an
integer axis chooses its increment to fit the range of the data into the range of the type, and
a floating point axis stores metres with the increment 1 by default.

A point that is not measured is ``NaN`` in a floating point type. An integer has no value for it,
so a file with integer axes lists the valid points in a separate file of bits. The ``x`` and ``y``
of a missing point are not kept then and read as ``NaN``; floating point axes keep them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from ..exceptions import X3pFormatError

__all__ = [
    "DATA_TYPES",
    "DataType",
    "DataTypeInfo",
    "decode_raw",
    "encode_raw",
    "get_data_type",
    "suggest_scale",
]


class DataType(StrEnum):
    """Storage type of the coordinates of one axis, as written in the ``DataType`` element.

    Members are named after the stored representation; their string value is
    the exact letter written into ``main.xml``.
    """

    INT16 = "I"
    INT32 = "L"
    FLOAT32 = "F"
    FLOAT64 = "D"


@dataclass(frozen=True)
class DataTypeInfo:
    """Description of one x3p coordinate storage type.

    :param type: ``DataType`` element value.
    :param dtype: Little-endian NumPy dtype used to store the raw values in a
        binary data file.
    """

    type: DataType
    dtype: np.dtype

    @property
    def is_integer(self) -> bool:
        """Whether the type is an integer type, which has no value for a missing point."""
        return bool(self.dtype.kind == "i")


#: All supported x3p data types, keyed by their ``DataType`` code.
DATA_TYPES: Final[dict[DataType, DataTypeInfo]] = {
    DataType.INT16: DataTypeInfo(DataType.INT16, np.dtype("<i2")),
    DataType.INT32: DataTypeInfo(DataType.INT32, np.dtype("<i4")),
    DataType.FLOAT32: DataTypeInfo(DataType.FLOAT32, np.dtype("<f4")),
    DataType.FLOAT64: DataTypeInfo(DataType.FLOAT64, np.dtype("<f8")),
}


def get_data_type(identifier: DataType) -> DataTypeInfo:
    """Look up the description of a supported x3p data type.

    :param identifier: A :class:`DataType`.
    :raises X3pFormatError: If it is not a known data type.

    >>> get_data_type(DataType.FLOAT32).dtype
    dtype('float32')
    >>> get_data_type(DataType.INT16).dtype
    dtype('int16')
    """
    try:
        return DATA_TYPES[identifier]
    except KeyError:
        raise X3pFormatError(f"Unknown x3p data type {identifier!r}") from None


def suggest_scale(values: np.ndarray, data_type: DataTypeInfo) -> float:
    """Pick the increment that stores ``values`` in ``data_type`` at the highest resolution.

    A floating point type stores metres directly, so the increment is ``1``.
    An integer type is given the smallest increment for which the largest
    magnitude still fits without overflow.

    :param values: Coordinates in metres; non-finite values are ignored.
    :param data_type: Target storage type.

    >>> suggest_scale(np.array([0.0, 1.0]), get_data_type(DataType.FLOAT64))
    1.0
    >>> suggest_scale(np.array([0.0, 32767e-6]), get_data_type(DataType.INT16))
    1e-06
    """
    if not data_type.is_integer:
        return 1.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    max_abs = float(np.abs(finite).max())
    if max_abs == 0.0:
        return 1.0
    limit = float(np.iinfo(data_type.dtype).max)
    return max_abs / limit


def encode_raw(values: np.ndarray, data_type: DataTypeInfo, increment: float) -> np.ndarray:
    """Convert coordinates in metres to the raw values stored in the file.

    :param values: Coordinates in metres; non-finite values mark missing
        points.
    :param data_type: Target storage type.
    :param increment: Scale factor; ``raw = values / increment``.
    :returns: Array of dtype ``data_type.dtype``. A missing point is ``NaN``
        for a floating point type, and ``0`` for an integer type, which has
        no such value and relies on the validity file instead.
    :raises X3pFormatError: If a finite value does not fit ``data_type``.
    """
    missing = ~np.isfinite(values)
    # 0.0 is only a safe placeholder: it keeps NaN out of np.rint()/astype(),
    # whose behaviour on NaN is undefined for integer dtypes.
    scaled = np.where(missing, 0.0, values / increment)
    if data_type.is_integer:
        rounded = np.rint(scaled)
        info = np.iinfo(data_type.dtype)
        if rounded.size and (rounded.min() < info.min or rounded.max() > info.max):
            raise X3pFormatError(
                f"Coordinate out of range for x3p data type {data_type.type.name} "
                f"({info.min}..{info.max}); use a larger increment"
            )
        return rounded.astype(data_type.dtype)
    with np.errstate(over="ignore"):
        raw = scaled.astype(data_type.dtype)
    if np.any(np.isinf(raw)):
        raise X3pFormatError(f"Coordinate out of range for x3p data type {data_type.type.name}")
    raw[missing] = np.nan
    return raw


def decode_raw(raw: np.ndarray, increment: float) -> np.ndarray:
    """Convert raw stored values to coordinates in metres.

    :param raw: Values as stored in the file.
    :param increment: Scale factor; ``metres = raw * increment``.
    :returns: ``float64`` array in which every non-finite stored value is
        ``NaN``.
    """
    values = raw.astype(np.float64) * increment
    values[~np.isfinite(values)] = np.nan
    return values
