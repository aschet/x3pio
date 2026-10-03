# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from x3pio import (
    Axis,
    AxisType,
    DataType,
    Header,
    X3pFormatError,
    validate_increment,
    validate_rotation,
)


def test_an_axis_is_described_by_enums() -> None:
    axis = Axis(axis_type=AxisType.ABSOLUTE, data_type=DataType.INT32)
    assert not axis.is_incremental
    assert Axis().is_incremental


def test_header_defaults_to_incremental_x_and_y_and_an_absolute_z_axis() -> None:
    header = Header()
    assert header.x.is_incremental
    assert not header.z.is_incremental


def test_header_equality_compares_the_axes() -> None:
    assert Header(x=Axis(increment=2.0)) != Header()


@pytest.mark.parametrize("increment", [0.0, -1e-6, float("nan"), float("inf")])
def test_validate_increment_rejects(increment: float) -> None:
    with pytest.raises(X3pFormatError, match="positive"):
        validate_increment(increment)


def test_validate_increment_accepts_positive() -> None:
    validate_increment(1e-9)


def _rotation_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_validate_rotation_accepts_rotations() -> None:
    validate_rotation(np.eye(3))
    validate_rotation(_rotation_z(0.3))


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(3) * 2,
        np.diag([1.0, 1.0, -1.0]),
        np.ones((3, 3)),
        np.eye(2),
        np.full((3, 3), np.nan),
    ],
)
def test_validate_rotation_rejects_everything_else(rotation: np.ndarray) -> None:
    with pytest.raises(X3pFormatError):
        validate_rotation(rotation)
