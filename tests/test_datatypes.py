# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from x3pio import DataType, X3pFormatError
from x3pio.model.datatypes import (
    DATA_TYPES,
    decode_raw,
    encode_raw,
    get_data_type,
    suggest_scale,
)


@pytest.mark.parametrize(
    ("data_type", "dtype"),
    [
        (DataType.INT16, "<i2"),
        (DataType.INT32, "<i4"),
        (DataType.FLOAT32, "<f4"),
        (DataType.FLOAT64, "<f8"),
    ],
)
def test_data_types_are_little_endian(data_type: DataType, dtype: str) -> None:
    assert DATA_TYPES[data_type].dtype == np.dtype(dtype)


def test_get_data_type_looks_up_a_member() -> None:
    for member in DataType:
        assert get_data_type(member).type is member


def test_get_data_type_rejects_unknown() -> None:
    with pytest.raises(X3pFormatError, match="Unknown x3p data type"):
        get_data_type("uint8")  # type: ignore[arg-type]


@pytest.mark.parametrize("data_type", [DataType.INT16, DataType.INT32])
def test_integer_roundtrip_rounds_to_the_increment(data_type: DataType) -> None:
    values = np.array([0.0, 1.2e-6, -3.4e-6])
    raw = encode_raw(values, get_data_type(data_type), 1e-7)
    np.testing.assert_allclose(decode_raw(raw, 1e-7), values, atol=1e-13)


def test_float_roundtrip_is_exact_for_float64() -> None:
    values = np.array([0.1, -2.5e-9, 1e12])
    raw = encode_raw(values, get_data_type(DataType.FLOAT64), 1.0)
    assert np.array_equal(decode_raw(raw, 1.0), values)


def test_encode_marks_missing_points() -> None:
    values = np.array([1.0, np.nan, np.inf])
    floats = encode_raw(values, get_data_type(DataType.FLOAT32), 1.0)
    assert np.isnan(floats[1:]).all()
    ints = encode_raw(values, get_data_type(DataType.INT16), 1.0)
    assert ints.tolist() == [1, 0, 0]


def test_decode_turns_every_non_finite_value_into_nan() -> None:
    raw = np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32)
    decoded = decode_raw(raw, 2.0)
    assert decoded[0] == 2.0
    assert np.isnan(decoded[1:]).all()


@pytest.mark.parametrize("data_type", [DataType.INT16, DataType.INT32])
def test_encode_rejects_integer_overflow(data_type: DataType) -> None:
    limit = np.iinfo(get_data_type(data_type).dtype).max
    with pytest.raises(X3pFormatError, match="out of range"):
        encode_raw(np.array([limit + 1.0]), get_data_type(data_type), 1.0)


def test_encode_accepts_the_integer_limits() -> None:
    info = np.iinfo(np.int16)
    raw = encode_raw(
        np.array([info.min, info.max], dtype=float), get_data_type(DataType.INT16), 1.0
    )
    assert raw.tolist() == [info.min, info.max]


def test_encode_rejects_float32_overflow() -> None:
    with pytest.raises(X3pFormatError, match="out of range"):
        encode_raw(np.array([1e39]), get_data_type(DataType.FLOAT32), 1.0)


def test_suggest_scale_is_one_for_floats() -> None:
    assert suggest_scale(np.array([1e-9, 5.0]), get_data_type(DataType.FLOAT32)) == 1.0


@pytest.mark.parametrize("data_type", [DataType.INT16, DataType.INT32])
def test_suggest_scale_fits_the_largest_magnitude(data_type: DataType) -> None:
    resolved = get_data_type(data_type)
    values = np.array([-3.7e-6, 1.1e-6, 9.9e-6])
    scale = suggest_scale(values, resolved)
    raw = encode_raw(values, resolved, scale)
    assert np.abs(raw).max() == np.iinfo(resolved.dtype).max


def test_suggest_scale_defaults_to_one_without_finite_values() -> None:
    resolved = get_data_type(DataType.INT16)
    assert suggest_scale(np.array([np.nan]), resolved) == 1.0
    assert suggest_scale(np.zeros(3), resolved) == 1.0


@pytest.mark.parametrize("data_type", [DataType.INT16, DataType.INT32])
def test_suggested_scale_never_overflows(data_type: DataType) -> None:
    resolved = get_data_type(data_type)
    limit = np.iinfo(resolved.dtype).max
    rng = np.random.default_rng(0)
    for max_abs in rng.uniform(1e-12, 1e3, 5000):
        raw = encode_raw(
            np.array([-max_abs, max_abs]), resolved, suggest_scale(np.array([max_abs]), resolved)
        )
        assert raw.max() == limit
