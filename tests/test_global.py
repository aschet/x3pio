# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The rank of the data, the global coordinate system, and the 2-D access to rotated grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import x3pio
from helpers import absolute, array_file, build, flat_points, shaped_points, stacked
from x3pio import (
    AxisType,
    CoordinateSystem,
    DataStorage,
    DataType,
    FeatureType,
    Header,
    Placement,
    PointCloud,
    Surface,
    X3pFile,
)

SAMPLES = Path(__file__).parent / "interop_samples" / "opengps"
ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
TILT = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]])


def _surface() -> Surface:
    return Surface.loads(build())


def _transformed(rotation: np.ndarray | None) -> Surface:
    return _surface().with_placement(Placement(rotation, offset=[0.25, -0.5, 3.0]))


def _same_points(first: X3pFile, second: X3pFile) -> None:
    np.testing.assert_allclose(flat_points(first), flat_points(second), atol=1e-15, equal_nan=True)


# --- the shape of the data is the one of the matrix ---------------------------------------------


def test_a_single_layer_is_a_matrix_with_one_layer() -> None:
    x3p = _surface()
    assert stacked(x3p).shape == (1, 2, 3)
    assert stacked(x3p).shape == (1, 2, 3)
    assert len(x3p.layers) == 1
    # surface is the 2-D view of the layer, not a copy
    assert stacked(x3p)[0].shape == (2, 3)
    assert np.shares_memory(stacked(x3p)[0], stacked(x3p))


def test_a_profile_is_layers_of_points() -> None:
    buffer = array_file(np.array([1.0, 2.0, 3.0]), x_scale=1.0).dumps()
    x3p = X3pFile.loads(buffer)
    assert stacked(x3p).shape == (1, 3)
    assert stacked(x3p)[0].shape == (3,)
    assert np.shares_memory(stacked(x3p)[0], stacked(x3p))


def test_several_layers_have_the_same_kind_of_shape() -> None:
    x3p = Surface.from_array(np.zeros((2, 3, 4)), x_scale=1.0, y_scale=1.0)
    again = Surface.loads(x3p.dumps())
    assert stacked(again).shape == (2, 3, 4)
    assert len(again.layers) == 2
    assert again.is_regular_grid
    assert flat_points(again).shape == (24, 3)


def test_the_rank_does_not_depend_on_the_number_of_layers() -> None:
    one = array_file(np.zeros((3, 4)), x_scale=1.0, y_scale=1.0)
    two = array_file(np.zeros((2, 3, 4)), x_scale=1.0, y_scale=1.0)
    assert stacked(one).ndim == stacked(two).ndim == 3


def test_a_point_cloud_has_no_layers() -> None:
    cloud = PointCloud.from_points(np.zeros((4, 3)))
    assert not hasattr(cloud, "num_layers")
    assert stacked(cloud).shape == (4,)


def test_two_dimensional_data_cannot_be_written_as_a_surface() -> None:
    x3p = x3pio.Surface.from_stacked(Header(), np.zeros((2, 3)))
    with pytest.raises(x3pio.X3pFormatError, match="must be 2-D"):
        x3p.dumps()


def test_profile_property_of_hand_built_data() -> None:
    x3p = x3pio.Profile.from_stacked(Header(), np.arange(4.0).reshape(1, 4))
    assert stacked(x3p)[0].tolist() == [0.0, 1.0, 2.0, 3.0]


# --- the global coordinate system ------------------------------------------------------------


def test_globalized_without_rotation_keeps_the_regular_grid() -> None:
    x3p = _transformed(None)
    global_ = x3p.globalized()

    assert global_.is_regular_grid
    assert not global_.placement.has_rotation
    assert global_.placement.offset[2] == 0.0
    np.testing.assert_allclose(stacked(global_)[0], stacked(x3p)[0] + 3.0, equal_nan=True)
    # An incremental axis keeps the offset that places its first column and row.
    assert global_.placement.offset[0] == 0.25
    assert global_.placement.offset[1] == -0.5
    _same_points(global_, x3p)
    assert flat_points(global_)[0].tolist() == pytest.approx([0.25, -0.5, 4e-6 + 3.0 - 3e-6])


def test_globalized_leaves_the_original_untouched() -> None:
    x3p = _transformed(ROTATION)
    before = flat_points(x3p)
    heights = stacked(x3p).copy()
    x3p.globalized()
    assert x3p.placement.has_rotation
    assert x3p.placement.offset[2] == 3.0
    assert np.array_equal(stacked(x3p), heights, equal_nan=True)
    assert np.array_equal(flat_points(x3p), before, equal_nan=True)


def test_globalized_with_a_rotation_gives_absolute_axes() -> None:
    x3p = _transformed(ROTATION)
    global_ = x3p.globalized()

    assert global_.placement.is_identity
    assert not global_.is_regular_grid
    for axis in (global_.header.x, global_.header.y, global_.header.z):
        assert (axis.axis_type, axis.data_type, axis.increment) == (
            AxisType.ABSOLUTE,
            DataType.FLOAT64,
            1.0,
        )
    assert stacked(global_, "x") is not None
    assert stacked(global_, "y") is not None
    assert (
        absolute(global_, "x").shape
        == absolute(global_, "y").shape
        == stacked(global_).shape
        == (1, 2, 3)
    )
    _same_points(global_, x3p)
    # The same cell of the rotated grid, as the grid of points shows.
    grid = shaped_points(x3p)
    np.testing.assert_allclose(absolute(global_, "x"), grid[..., 0], equal_nan=True)
    np.testing.assert_allclose(absolute(global_, "y"), grid[..., 1], equal_nan=True)
    np.testing.assert_allclose(stacked(global_), grid[..., 2], equal_nan=True)


def test_globalized_with_a_tilt_changes_the_heights() -> None:
    x3p = _surface().with_placement(Placement(TILT))
    global_ = x3p.globalized()
    assert not np.allclose(stacked(global_), stacked(x3p), equal_nan=True)
    _same_points(global_, x3p)


@pytest.mark.parametrize("storage", list(DataStorage))
@pytest.mark.parametrize("rotation", [None, ROTATION, TILT])
def test_globalized_survives_writing_and_reading(
    storage: DataStorage, rotation: np.ndarray | None
) -> None:
    x3p = _transformed(rotation)
    global_ = x3p.globalized()
    again = X3pFile.loads(global_.dumps(storage=storage))
    _same_points(again, x3p)


def test_globalized_is_idempotent() -> None:
    global_ = _transformed(TILT).globalized()
    again = global_.globalized()
    assert again == global_


def test_globalized_without_offsets_or_rotation_changes_nothing() -> None:
    x3p = _surface()
    assert x3p.globalized() == x3p


def test_globalized_with_an_identity_rotation_is_the_trivial_case() -> None:
    x3p = _transformed(np.eye(3))
    global_ = x3p.globalized()
    assert global_.is_regular_grid
    assert not global_.placement.has_rotation
    _same_points(global_, x3p)


def test_globalized_of_an_integer_axis_with_an_offset_becomes_float() -> None:
    x3p = X3pFile.loads(build(z_type=DataType.INT16))
    x3p = x3p.with_placement(Placement.translation(0.0, 0.0, 1e-6 / 3))
    global_ = x3p.globalized()
    assert global_.header.z.data_type is DataType.FLOAT64
    assert global_.header.z.increment == 1.0
    _same_points(X3pFile.loads(global_.dumps()), x3p)


def test_globalized_of_an_integer_axis_without_offset_keeps_its_type() -> None:
    x3p = X3pFile.loads(build(z_type=DataType.INT16))
    assert x3p.globalized().header.z.data_type is DataType.INT16


def test_globalized_keeps_unmeasured_points_unmeasured() -> None:
    global_ = _transformed(TILT).globalized()
    assert np.isnan(stacked(global_)[0, 0, 2])
    assert np.isnan(flat_points(global_)[2]).all()


def test_globalized_of_a_point_cloud() -> None:
    cloud = PointCloud.from_points(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    cloud = cloud.with_placement(Placement(ROTATION, offset=[1.0, 0.0, -2.0]))
    global_ = cloud.globalized()
    assert global_.feature_type is FeatureType.POINT_CLOUD
    assert stacked(global_).shape == (2,)
    _same_points(global_, cloud)
    again = X3pFile.loads(global_.dumps())
    _same_points(again, cloud)


def test_globalized_of_samples_with_offsets_and_absolute_axes() -> None:
    for name in ("ISO5436-sample2.x3p", "ISO5436-sample3.x3p", "ISO5436-sample4.x3p"):
        x3p = x3pio.read(SAMPLES / name)
        _same_points(x3p.globalized(), x3p)
    sample3 = x3pio.read(SAMPLES / "ISO5436-sample3.x3p").globalized()
    assert sample3.header.y.data_type is DataType.FLOAT64  # was int16 with an offset
    assert sample3.header.x.is_incremental
    assert sample3.placement.offset[0] == 0.1


# --- 2-D access that also works with a rotation ------------------------------------------------


def test_point_grid_has_a_point_per_row_and_column() -> None:
    x3p = _surface()
    grid = shaped_points(x3p)
    assert grid.shape == (1, 2, 3, 3)
    np.testing.assert_allclose(grid[..., 2], stacked(x3p), equal_nan=True)
    assert grid[0, 1, 2, :2].tolist() == pytest.approx([2e-6, 2e-6])
    assert np.array_equal(grid.reshape(-1, 3), flat_points(x3p), equal_nan=True)


def test_point_grid_applies_rotation_and_offset_unless_told_not_to() -> None:
    x3p = _transformed(ROTATION)
    assert np.allclose(shaped_points(x3p).reshape(-1, 3), flat_points(x3p), equal_nan=True)
    view = shaped_points(x3p, coordinate_system=CoordinateSystem.VIEW)
    assert np.array_equal(
        view.reshape(-1, 3),
        flat_points(x3p, coordinate_system=CoordinateSystem.VIEW),
        equal_nan=True,
    )
    assert not np.allclose(view[..., :2], shaped_points(x3p)[..., :2])


def test_point_grid_of_a_tilted_surface_is_no_regular_grid() -> None:
    heights = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 4.0]]) * 1e-3
    x3p = Surface.from_array(heights, x_scale=1e-3, y_scale=1e-3)
    x3p = x3p.with_placement(Placement(TILT))
    row_distance = np.diff(shaped_points(x3p)[0, ..., 1], axis=0)[0]
    # In a regular grid, all points of a column have the same distance to the next row. The tilt
    # moves each point in y by its height, so the distance differs from column to column.
    assert len({round(float(value), 12) for value in row_distance}) == 3


def test_the_points_hold_every_layer_and_every_type_has_them() -> None:
    stack = array_file(np.zeros((2, 2, 2)), x_scale=1.0, y_scale=1.0)
    assert isinstance(stack, x3pio.Surface)
    assert shaped_points(stack).shape == (2, 2, 2, 3)
    assert shaped_points(PointCloud.from_points(np.zeros((3, 3)))).shape == (3, 3)
