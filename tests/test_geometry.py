# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The coordinate arithmetic behind the file types, as functions of arrays."""

from __future__ import annotations

import numpy as np
import pytest

from x3pio import Axis, AxisType, CoordinateSystem, DataType, Header, Placement, X3pFormatError
from x3pio.model.arithmetic import (
    axis_values,
    centered,
    globalize,
    resolve_coordinate_system,
    view_points,
)

_C, _S = np.cos(0.4), np.sin(0.4)
ROTATION = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
OFFSET = np.array([0.01, -0.02, 0.003])
PLACEMENT = Placement(ROTATION, OFFSET)


def test_an_incremental_axis_counts_along_its_index() -> None:
    axis = Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 2.0)
    x = axis_values(axis, None, (2, 3, 4), -1)
    y = axis_values(axis, None, (2, 3, 4), -2)
    assert x.shape == y.shape == (2, 3, 4)
    assert x[1, 2].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert y[1, :, 3].tolist() == [0.0, 2.0, 4.0]


def test_an_incremental_axis_of_a_list_counts_the_position() -> None:
    axis = Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 0.5)
    assert axis_values(axis, None, (4,), -1).tolist() == [0.0, 0.5, 1.0, 1.5]


def test_an_absolute_axis_gives_its_coordinates_and_needs_them() -> None:
    axis = Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0)
    stored = np.arange(6.0).reshape(1, 2, 3)
    assert axis_values(axis, stored, stored.shape, -1) is stored
    with pytest.raises(X3pFormatError, match="needs its coordinates"):
        axis_values(axis, None, stored.shape, -1)


def test_the_points_of_a_matrix_come_in_storage_order() -> None:
    header = Header(
        x=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 1.0),
        y=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 10.0),
        z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
    )
    heights = np.arange(6.0).reshape(1, 2, 3)
    points = view_points(header, heights, None, None)
    assert points.shape == (6, 3)
    assert points[:, 0].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    assert points[:, 1].tolist() == [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
    assert points[:, 2].tolist() == heights.ravel().tolist()


def test_no_placement_means_the_two_systems_coincide() -> None:
    placement, is_global = resolve_coordinate_system(None, None)
    assert placement.is_identity
    assert not is_global
    assert resolve_coordinate_system(None, CoordinateSystem.GLOBAL)[
        0
    ].is_identity  # the coordinate system does not matter then


def test_a_placement_needs_the_system_of_the_coordinates() -> None:
    with pytest.raises(TypeError, match="coordinate_system"):
        resolve_coordinate_system(PLACEMENT, None)
    assert resolve_coordinate_system(PLACEMENT, CoordinateSystem.GLOBAL) == (PLACEMENT, True)
    assert resolve_coordinate_system(PLACEMENT, CoordinateSystem.VIEW) == (PLACEMENT, False)


def test_globalizing_leaves_its_arguments_alone_and_shares_nothing() -> None:
    header = Header(
        x=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 1.0),
        y=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 1.0),
        z=Axis(AxisType.ABSOLUTE, DataType.INT16, 1e-6),
    )
    placement = Placement(offset=[2.0, 0.0, 1.0])
    z = np.zeros((1, 2, 2))
    new_header, new_placement, new_z, x, y = globalize(header, placement, z, None, None)
    assert header.z.data_type is DataType.INT16
    assert new_header.z.data_type is DataType.FLOAT64  # shifted values are no multiples any more
    assert new_placement.offset.tolist() == [2.0, 0.0, 0.0]  # the first column stays at x = 2
    assert new_z.tolist() == [[[1.0, 1.0], [1.0, 1.0]]]
    assert x is None
    assert y is None
    for name in ("x", "y", "z"):
        assert getattr(new_header, name) is not getattr(header, name)


def test_globalizing_a_rotated_grid_gives_absolute_axes() -> None:
    header = Header(
        x=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 1.0),
        y=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 1.0),
        z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
    )
    new_header, new_placement, z, x, y = globalize(
        header, PLACEMENT, np.zeros((1, 2, 3)), None, None
    )
    assert new_placement.is_identity
    assert not new_header.x.is_incremental
    assert x is not None
    assert y is not None
    assert x.shape == y.shape == z.shape == (1, 2, 3)
    assert header.x.is_incremental


def test_globalizing_keeps_the_global_points() -> None:
    header = Header(
        x=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 1e-3),
        y=Axis(AxisType.INCREMENTAL, DataType.FLOAT64, 2e-3),
        z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
    )
    z = np.arange(6.0).reshape(1, 2, 3) * 1e-6
    for placement in (PLACEMENT, Placement(offset=OFFSET)):
        before = placement.apply(view_points(header, z, None, None))
        new_header, new_placement, new_z, new_x, new_y = globalize(header, placement, z, None, None)
        after = new_placement.apply(view_points(new_header, new_z, new_x, new_y))
        np.testing.assert_allclose(after, before, atol=1e-15)


def _absolute_header() -> Header:
    return Header(
        x=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
        y=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
        z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
    )


def test_centering_shifts_by_the_middle_of_each_range_and_keeps_the_global_points() -> None:
    x = np.array([10.0, 12.0, 11.0])
    y = np.array([0.0, 2.0, np.nan])
    z = np.array([5.0, 7.0, 6.0])
    header = _absolute_header()
    for placement in (Placement(), PLACEMENT):
        moved, new_z, new_x, new_y = centered(header, placement, z, x, y)
        assert new_x is not None
        assert new_y is not None
        assert new_x.tolist() == [-1.0, 1.0, 0.0]
        assert new_z.tolist() == [-1.0, 1.0, 0.0]
        np.testing.assert_array_equal(new_y[:2], [-1.0, 1.0])
        before = placement.apply(view_points(header, z, x, y))
        after = moved.apply(view_points(header, new_z, new_x, new_y))
        np.testing.assert_allclose(after, before, atol=1e-12)


def test_centering_leaves_an_incremental_axis_and_values_that_are_not_finite_alone() -> None:
    header = Header(z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0))
    z = np.array([[[1.0, 3.0], [np.nan, np.nan]]])
    placement = Placement(offset=[1.0, 2.0, 3.0])
    moved, new_z, new_x, new_y = centered(header, placement, z, None, None)
    assert new_x is None
    assert new_y is None
    assert moved.offset.tolist() == [1.0, 2.0, 5.0]
    assert new_z[0, 0].tolist() == [-1.0, 1.0]
    nothing = centered(header, placement, np.full((1, 1, 2), np.nan), None, None)
    assert nothing[0] == placement
