# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The three types of file: a profile, a surface and a point cloud, and their layers."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import x3pio
from helpers import absolute, build, flat_points, shaped_points, stacked, unzip
from x3pio import (
    Axis,
    AxisType,
    CoordinateSystem,
    DataStorage,
    DataType,
    FeatureType,
    Header,
    Layer,
    Placement,
    PointCloud,
    PointLayer,
    Profile,
    ProfileLayer,
    Surface,
    SurfaceLayer,
    X3pFile,
    X3pFormatError,
)
from x3pio.cli import main

ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _blob(file: X3pFile) -> io.BytesIO:
    return io.BytesIO(file.dumps())


# --- the base is not a file -----------------------------------------------------------------------


def test_the_base_type_cannot_be_made() -> None:
    with pytest.raises(TypeError, match="abstract"):
        X3pFile([])  # type: ignore[abstract]


def test_every_type_knows_its_feature_type() -> None:
    assert Profile.from_array(np.zeros(3), x_scale=1.0).feature_type is FeatureType.PROFILE
    assert (
        Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0).feature_type
        is FeatureType.SURFACE
    )
    assert PointCloud.from_points(np.zeros((2, 3))).feature_type is FeatureType.POINT_CLOUD


# --- making the types -----------------------------------------------------------------------------


def test_write_picks_the_type_by_the_number_of_dimensions() -> None:
    cases = [
        (np.zeros(3), {}, Profile),
        (np.zeros((2, 3)), {"y_scale": 1.0}, Surface),
        (np.zeros((2, 2, 3)), {"y_scale": 1.0}, Surface),
    ]
    for data, scales, kind in cases:
        buffer = io.BytesIO()
        x3pio.write(buffer, data, x_scale=1.0, **scales)  # type: ignore[arg-type]
        assert isinstance(x3pio.read(io.BytesIO(buffer.getvalue())), kind)
    with pytest.raises(X3pFormatError, match="1-D profile"):
        x3pio.write(io.BytesIO(), np.zeros((1, 1, 1, 1)), x_scale=1.0)


def test_a_profile_is_one_or_two_dimensional() -> None:
    assert stacked(Profile.from_array(np.zeros(4), x_scale=1.0)).shape == (1, 4)
    assert stacked(Profile.from_array(np.zeros((3, 4)), x_scale=1.0)).shape == (3, 4)
    with pytest.raises(X3pFormatError, match="layers"):
        Profile.from_array(np.zeros((1, 1, 4)), x_scale=1.0)


def test_a_surface_is_two_or_three_dimensional() -> None:
    assert stacked(Surface.from_array(np.zeros((2, 4)), x_scale=1.0, y_scale=1.0)).shape == (
        1,
        2,
        4,
    )
    assert stacked(Surface.from_array(np.zeros((3, 2, 4)), x_scale=1.0, y_scale=1.0)).shape == (
        3,
        2,
        4,
    )
    with pytest.raises(X3pFormatError, match="rows"):
        Surface.from_array(np.zeros(4), x_scale=1.0, y_scale=1.0)
    with pytest.raises(TypeError, match="y_scale"):
        Surface.from_array(np.zeros((2, 4)), x_scale=1.0)


def test_a_profile_takes_the_x_increment_for_the_y_axis_it_does_not_use() -> None:
    assert Profile.from_array(np.zeros(3), x_scale=2.0).header.y.increment == 2.0
    with pytest.raises(TypeError, match="y_scale"):
        Profile.from_array(np.zeros(3), x_scale=2.0, y_scale=5.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="y_scale"):
        Profile.from_layers([np.zeros(3)], x_scale=2.0, y_scale=5.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="no y_scale"):
        x3pio.write(io.BytesIO(), np.zeros(3), x_scale=2.0, y_scale=5.0)


# --- reading --------------------------------------------------------------------------------------


def test_read_returns_the_type_the_file_has() -> None:
    profile = Profile.from_array(np.zeros((2, 5)), x_scale=1e-6)
    surface = Surface.from_array(np.zeros((3, 4)), x_scale=1e-6, y_scale=1e-6)
    cloud = PointCloud.from_points(np.arange(12.0).reshape(4, 3))
    for original in (profile, surface, cloud):
        again = x3pio.read(_blob(original))
        assert type(again) is type(original)
        assert again == original or isinstance(original, PointCloud)  # rounding of the cloud
        np.testing.assert_allclose(stacked(again), stacked(original))


def test_a_type_asked_for_has_to_be_the_one_the_file_has() -> None:
    cloud = _blob(PointCloud.from_points(np.zeros((2, 3)))).getvalue()
    with pytest.raises(X3pFormatError, match="holds a point cloud, not a surface"):
        Surface.loads(cloud)
    with pytest.raises(X3pFormatError, match="not a profile"):
        Profile.open(io.BytesIO(cloud))
    assert isinstance(PointCloud.loads(cloud), PointCloud)
    assert isinstance(X3pFile.loads(cloud), PointCloud)


def test_a_profile_file_with_a_single_row_is_a_profile_of_layers() -> None:
    again = x3pio.read(_blob(Profile.from_array(np.arange(6.0).reshape(2, 3), x_scale=1.0)))
    assert isinstance(again, Profile)
    assert stacked(again).tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


# --- layers ---------------------------------------------------------------------------------------


def test_a_surface_has_its_layers_as_typed_views() -> None:
    stack = np.arange(24.0).reshape(2, 3, 4) * 1e-6
    surface = Surface.from_array(stack, x_scale=1e-6, y_scale=1e-6)
    layers = surface.layers
    assert len(layers) == 2
    assert all(isinstance(layer, SurfaceLayer) for layer in layers)
    for index, layer in enumerate(layers):
        assert layer.z.shape == (3, 4)
        assert np.array_equal(layer.z, stack[index])
        assert layer.x is None
        assert layer.y is None


def test_a_profile_has_its_layers_as_typed_views() -> None:
    profile = Profile.from_array(np.arange(6.0).reshape(2, 3), x_scale=1.0)
    layers = profile.layers
    assert len(layers) == 2
    assert all(isinstance(layer, ProfileLayer) for layer in layers)
    assert layers[1].z.tolist() == [3.0, 4.0, 5.0]


def test_the_layers_of_absolute_axes_carry_their_coordinates() -> None:
    header = Header(x=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0), y=Axis(AxisType.ABSOLUTE))
    z = np.arange(8.0).reshape(2, 2, 2)
    surface = Surface.from_stacked(header, z, x=z * 2, y=z * 3)
    for index, layer in enumerate(surface.layers):
        assert layer.x is not None
        assert layer.y is not None
        assert np.array_equal(layer.x, z[index] * 2)
        assert np.array_equal(layer.y, z[index] * 3)
    profile = Profile.from_stacked(header, z[:, 0], x=z[:, 0] * 2, y=z[:, 0] * 3)
    assert profile.layers[1].x is not None
    assert profile.layers[1].x.tolist() == [8.0, 10.0]


def test_a_layer_knows_its_measured_points() -> None:
    surface = Surface.from_array(np.array([[1.0, np.nan]]), x_scale=1.0, y_scale=1.0)
    assert surface.layers[0].valid.tolist() == [[True, False]]
    profile = Profile.from_array(np.array([np.nan, 2.0]), x_scale=1.0)
    assert profile.layers[0].valid.tolist() == [False, True]


def test_writing_to_a_layer_changes_the_file() -> None:
    surface = Surface.from_array(np.zeros((1, 2, 2)), x_scale=1.0, y_scale=1.0)
    surface.layers[0].z[0, 1] = 5.0
    assert stacked(surface)[0, 0, 1] == 5.0


# --- the axes -------------------------------------------------------------------------------------


def test_a_profile_has_an_x_axis_only() -> None:
    profile = Profile.from_array(np.zeros(3), x_scale=0.5)
    assert profile.layers[0].x_values.tolist() == [0.0, 0.5, 1.0]
    assert not hasattr(profile, "y_axis")
    absolute = Profile.from_stacked(
        Header(x=Axis(AxisType.ABSOLUTE)), np.zeros((1, 2)), x=np.zeros((1, 2))
    )
    with pytest.raises(ValueError, match="x_values"):
        _ = absolute.layers[0].x_values


def test_a_surface_with_an_absolute_axis_has_no_such_axis() -> None:
    header = Header(y=Axis(AxisType.ABSOLUTE))
    surface = Surface.from_stacked(header, np.zeros((1, 2, 2)), y=np.zeros((1, 2, 2)))
    assert not surface.is_regular_grid
    with pytest.raises(ValueError, match="y_values"):
        _ = surface.layers[0].y_values
    assert surface.layers[0].x_values.tolist() == [0.0, 1.0]


def test_the_spacing_of_a_regular_layer() -> None:
    surface = Surface.from_array(np.zeros((2, 3)), x_scale=2.0, y_scale=0.5)
    assert surface.layers[0].spacing == (2.0, 0.5)
    assert Profile.from_array(np.zeros(3), x_scale=0.25).layers[0].spacing == 0.25


def test_the_extent_of_a_surface_is_the_edges_of_its_cells() -> None:
    layer = Surface.from_array(np.zeros((2, 3)), x_scale=2.0, y_scale=0.5).layers[0]
    assert layer.extent == (-1.0, 5.0, -0.25, 0.75)
    x_values, y_values = layer.x_values, layer.y_values
    assert (layer.extent[0], layer.extent[1]) == (x_values[0] - 1.0, x_values[-1] + 1.0)
    assert (layer.extent[2], layer.extent[3]) == (y_values[0] - 0.25, y_values[-1] + 0.25)


def test_every_kind_of_layer_tells_whether_it_is_a_regular_grid() -> None:
    assert Surface.from_array(np.zeros((2, 3)), x_scale=1.0, y_scale=1.0).layers[0].is_regular_grid
    assert Profile.from_array(np.zeros(3), x_scale=1.0).layers[0].is_regular_grid
    assert not Surface.from_points(np.zeros((2, 2, 3))).layers[0].is_regular_grid
    assert not Profile.from_points(np.zeros((3, 3))).layers[0].is_regular_grid
    assert not PointCloud.from_points(np.zeros((2, 3))).layer.is_regular_grid
    # The layer says what spacing and extent need: they work exactly when it is true
    for layer in (
        Surface.from_array(np.zeros((2, 3)), x_scale=1.0, y_scale=1.0).layers[0],
        Surface.from_points(np.zeros((2, 2, 3))).layers[0],
    ):
        if layer.is_regular_grid:
            assert layer.spacing == (1.0, 1.0)
        else:
            with pytest.raises(ValueError, match="spacing"):
                _ = layer.spacing


def test_spacing_and_extent_need_regular_axes() -> None:
    irregular = Surface.from_points(np.zeros((2, 2, 3))).layers[0]
    with pytest.raises(ValueError, match="spacing"):
        _ = irregular.spacing
    with pytest.raises(ValueError, match="spacing"):
        _ = irregular.extent
    absolute = Profile.from_points(np.zeros((2, 3))).layers[0]
    with pytest.raises(ValueError, match="spacing"):
        _ = absolute.spacing


# --- the rotation of a profile --------------------------------------------------------------------


def test_a_rotated_profile_becomes_absolute_in_every_layer() -> None:
    profile = Profile.from_array(np.arange(6.0).reshape(2, 3), x_scale=1.0)
    profile = profile.with_placement(Placement(ROTATION))
    global_ = profile.globalized()
    assert isinstance(global_, Profile)
    assert stacked(global_, "x") is not None
    assert stacked(global_, "y") is not None
    assert (
        absolute(global_, "x").shape
        == absolute(global_, "y").shape
        == stacked(global_).shape
        == (2, 3)
    )
    np.testing.assert_allclose(flat_points(global_), flat_points(profile), atol=1e-15)
    again = x3pio.read(_blob(global_))
    assert isinstance(again, Profile)
    np.testing.assert_allclose(flat_points(again), flat_points(profile), atol=1e-12)


# --- the command line -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file", "expected"),
    [
        (
            Profile.from_array(np.zeros((2, 5)), x_scale=1e-6),
            ["SizeX           = 5", "SizeY", "SizeZ           = 2"],
        ),
        (
            Surface.from_array(np.zeros((3, 4)), x_scale=1e-6, y_scale=1e-6),
            ["SizeX           = 4", "SizeY           = 3", "SizeZ           = 1"],
        ),
    ],
    ids=["profile", "surface"],
)
def test_info_shows_the_size_of_a_profile_and_a_surface(
    file: X3pFile,
    expected: list[str],
    tmp_path: pytest.TempPathFactory,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "a.x3p"  # type: ignore[operator]
    file.save(path)
    assert main(["info", str(path)]) == 0
    output = capsys.readouterr().out
    assert all(line in output for line in expected)


def test_a_file_written_by_x3pio_names_the_feature_type() -> None:
    for original, name in (
        (Profile.from_array(np.zeros(3), x_scale=1.0), "PRF"),
        (Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0), "SUR"),
        (PointCloud.from_points(np.zeros((2, 3))), "PCL"),
    ):
        assert f"<FeatureType>{name}</FeatureType>" in unzip(original.dumps())["main.xml"].decode()


def test_the_type_survives_a_conversion_of_the_revision() -> None:
    for original in (
        Profile.from_array(np.zeros((2, 3)), x_scale=1.0),
        Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0),
        PointCloud.from_points(np.zeros((2, 3))),
    ):
        converted = original.with_revision(x3pio.Revision.ISO5436_2000)
        assert type(converted) is type(original)
        assert type(x3pio.read(_blob(converted))) is type(original)


def test_build_gives_a_surface() -> None:
    assert isinstance(x3pio.X3pFile.loads(build()), Surface)


# --- absolute axes: an irregular surface and a profile along any path -------------------------


def _irregular_grid() -> np.ndarray:
    x = np.array([[0.0, 1.0, 2.5], [0.0, 1.5, 3.0]]) * 1e-3
    y = np.array([[0.0, 0.0, 0.1], [1.0, 1.0, 1.2]]) * 1e-3
    z = np.arange(6.0).reshape(2, 3) * 1e-6
    return np.stack([x, y, z], axis=-1)


def test_an_irregular_surface_is_made_from_its_points() -> None:
    grid = _irregular_grid()
    surface = Surface.from_points(grid)
    assert stacked(surface).shape == (1, 2, 3)
    assert not surface.is_regular_grid
    assert surface.header.x.axis_type is AxisType.ABSOLUTE
    assert surface.header.y.axis_type is AxisType.ABSOLUTE
    layer = surface.layers[0]
    assert layer.x is not None
    assert layer.y is not None
    np.testing.assert_array_equal(layer.x, grid[..., 0])
    np.testing.assert_array_equal(layer.y, grid[..., 1])
    np.testing.assert_array_equal(layer.z, grid[..., 2])
    np.testing.assert_array_equal(shaped_points(surface)[0], grid)


@pytest.mark.parametrize("storage", list(x3pio.DataStorage))
@pytest.mark.parametrize("data_type", list(DataType))
def test_an_irregular_surface_survives_every_type_and_storage(
    storage: x3pio.DataStorage, data_type: DataType
) -> None:
    grid = _irregular_grid()
    surface = Surface.from_points(grid, data_type=data_type, storage=storage)
    again = x3pio.read(_blob(surface))
    assert isinstance(again, Surface)
    # An integer type stores the largest coordinate in 15 bits, so half a step is about 5e-8.
    np.testing.assert_allclose(shaped_points(again)[0], grid, atol=1e-7, rtol=0)


def test_a_surface_of_layers_from_a_grid_of_points() -> None:
    lifted = _irregular_grid() + np.array([0.0, 0.0, 1e-3])
    grid = np.stack([_irregular_grid(), lifted])
    surface = Surface.from_points(grid)
    assert stacked(surface).shape == (2, 2, 3)
    np.testing.assert_array_equal(shaped_points(surface), grid)


def test_the_grid_is_the_inverse_of_the_point_grid_of_a_surface() -> None:
    regular = Surface.from_array(np.arange(6.0).reshape(2, 3) * 1e-6, x_scale=1e-3, y_scale=2e-3)
    again = Surface.from_points(shaped_points(regular, coordinate_system=CoordinateSystem.VIEW))
    np.testing.assert_allclose(flat_points(again), flat_points(regular), atol=1e-15)
    assert not again.is_regular_grid


def test_a_point_of_a_grid_that_is_not_measured_stays_missing() -> None:
    grid = _irregular_grid()
    grid[0, 1, 2] = np.nan
    again = x3pio.read(_blob(Surface.from_points(grid)))
    assert isinstance(again, Surface)
    assert again.layers[0].valid.tolist() == [[True, False, True], [True, True, True]]
    assert stacked(again)[0, 0, 1] != stacked(again)[0, 0, 1]  # NaN


def test_a_grid_in_the_global_system_is_stored_in_the_view_system() -> None:
    grid = _irregular_grid()
    placement = Placement(ROTATION, offset=[0.5, 0.25, 1.0])
    surface = Surface.from_points(
        grid, placement=placement, coordinate_system=CoordinateSystem.GLOBAL
    )
    assert surface.placement == placement
    np.testing.assert_allclose(shaped_points(surface)[0], grid, atol=1e-15)
    stored = Surface.from_points(grid, placement=placement, coordinate_system=CoordinateSystem.VIEW)
    np.testing.assert_array_equal(stored.layers[0].z, grid[..., 2])


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 2), (1, 1, 2, 3, 3), (3,)])
def test_a_grid_of_points_has_the_right_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(X3pFormatError, match="grid of points"):
        Surface.from_points(np.zeros(shape))


def test_a_profile_may_follow_any_path() -> None:
    path = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 3.0]]) * 1e-3
    profile = Profile.from_points(path)
    assert stacked(profile).shape == (1, 3)
    assert profile.header.x.axis_type is AxisType.ABSOLUTE
    layer = profile.layers[0]
    assert layer.x is not None
    assert layer.y is not None
    assert layer.x.tolist() == path[:, 0].tolist()
    assert layer.y.tolist() == path[:, 1].tolist()
    again = x3pio.read(_blob(profile))
    assert isinstance(again, Profile)
    np.testing.assert_allclose(flat_points(again), path, atol=1e-15)


def test_a_profile_of_layers_from_points() -> None:
    layers = np.stack([np.zeros((4, 3)), np.ones((4, 3))])
    profile = Profile.from_points(layers)
    assert stacked(profile).shape == (2, 4)
    np.testing.assert_array_equal(flat_points(profile), layers.reshape(-1, 3))


@pytest.mark.parametrize("shape", [(3,), (2, 4), (1, 1, 3, 3)])
def test_the_points_of_a_profile_have_the_right_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(X3pFormatError, match="Points must be"):
        Profile.from_points(np.zeros(shape))


def test_the_absolute_constructors_take_the_revision() -> None:
    revision = x3pio.Revision.ISO5436_2000
    assert Surface.from_points(_irregular_grid(), revision=revision).revision is revision
    assert Profile.from_points(np.zeros((2, 3)), revision=revision).revision is revision


# --- layers: a sequence, made from layers, and the points of one layer ---------------------------


def test_the_layers_are_the_same_objects_every_time() -> None:
    stack = Surface.from_array(np.arange(24.0).reshape(3, 2, 4), x_scale=1.0, y_scale=1.0)
    layers = stack.layers
    assert isinstance(layers, tuple)
    assert len(layers) == 3
    assert stack.layers[0] is layers[0]
    assert stack.layers is layers
    assert layers[-1].z[0, 0] == 16.0
    assert [layer.z[0, 0] for layer in layers[1:]] == [8.0, 16.0]
    layers[0].z[0, 0] = -1.0  # the layer is the file's data
    assert stack.layers[0].z[0, 0] == -1.0
    assert stacked(stack)[0, 0, 0] == -1.0


def test_layers_are_equal_by_value_and_not_hashable() -> None:
    first = Surface.from_array(np.array([[1.0, np.nan]]), x_scale=1.0, y_scale=1.0).layers[0]
    same = Surface.from_array(np.array([[1.0, np.nan]]), x_scale=1.0, y_scale=1.0).layers[0]
    assert first is not same
    assert first == same  # NaN equals NaN
    assert first != Surface.from_array(np.array([[1.0, 2.0]]), x_scale=1.0, y_scale=1.0).layers[0]
    assert (
        first != Surface.from_array(np.array([[1.0, np.nan]]), x_scale=2.0, y_scale=1.0).layers[0]
    )
    moved = Surface.from_array(
        np.array([[1.0, np.nan]]),
        x_scale=1.0,
        y_scale=1.0,
        placement=Placement.translation(1.0, 0.0, 0.0),
        coordinate_system=CoordinateSystem.VIEW,
    ).layers[0]
    assert first != moved
    assert first != Profile.from_array(np.array([1.0, np.nan]), x_scale=1.0).layers[0]
    assert first != "layer"
    with pytest.raises(TypeError, match="unhashable"):
        hash(first)


def test_the_header_of_a_layer_cannot_be_changed() -> None:
    layer = Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0).layers[0]
    with pytest.raises(FrozenInstanceError):
        layer.header.z = Axis()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        layer.header.x.increment = 2.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        layer.placement = Placement()  # type: ignore[misc]


def test_the_axes_and_the_placement_of_a_file_are_those_of_its_layers() -> None:
    surface = Surface.from_array(np.zeros((2, 1, 2)), x_scale=1.0, y_scale=1.0)
    assert surface.header is surface.layers[0].header
    assert surface.layers[1].header is surface.header
    assert surface.placement is surface.layers[0].placement
    with pytest.raises(AttributeError):
        surface.header = Header()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        surface.placement = Placement()  # type: ignore[misc]


def test_a_file_is_made_of_layers_that_it_keeps() -> None:
    layers = Surface.from_array(np.zeros((2, 1, 2)), x_scale=1.0, y_scale=1.0).layers
    surface = Surface(layers)
    assert surface.layers == layers
    assert surface.layers[0] is layers[0]  # not copied
    assert surface.metadata is None
    assert surface.storage is DataStorage.BINARY


@pytest.mark.parametrize(
    ("make", "error", "message"),
    [
        (lambda layer, other: [], X3pFormatError, "at least one layer"),
        (lambda layer, other: [layer, other[1]], X3pFormatError, "same shape"),
        (lambda layer, other: [layer, other[2]], X3pFormatError, "same axes and placement"),
        (lambda layer, other: [layer, other[3]], X3pFormatError, "absolute axes, or none"),
        (lambda layer, other: [other[4]], TypeError, "SurfaceLayer objects"),
    ],
)
def test_the_layers_of_a_file_have_to_fit_together(
    make: Callable[[SurfaceLayer, list[Layer]], list[Layer]], error: type[Exception], message: str
) -> None:
    layer = Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0).layers[0]
    other = [
        layer,
        Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0).layers[0],
        Surface.from_array(np.zeros((1, 2)), x_scale=2.0, y_scale=1.0).layers[0],
        SurfaceLayer(layer.z, np.zeros((1, 2)), header=layer.header),
        Profile.from_array(np.zeros(2), x_scale=1.0).layers[0],
    ]
    with pytest.raises(error, match=message):
        Surface(make(layer, other))


def test_a_point_cloud_has_exactly_one_layer() -> None:
    layer = PointCloud.from_points(np.zeros((2, 3))).layer
    with pytest.raises(X3pFormatError, match="exactly one layer"):
        PointCloud([layer, layer])


def test_stacked_arrays_make_layers_that_view_them() -> None:
    z = np.arange(8.0).reshape(2, 2, 2)
    surface = Surface.from_stacked(Header(), z)
    z[1, 0, 0] = -5.0
    assert surface.layers[1].z[0, 0] == -5.0
    with pytest.raises(X3pFormatError, match="same number of layers"):
        Surface.from_stacked(Header(), z, x=z[:1])
    with pytest.raises(X3pFormatError, match="must be arrays"):
        Surface.from_stacked(Header(), np.float64(1.0))  # type: ignore[arg-type]


def test_a_surface_is_made_from_its_layers() -> None:
    scans = [np.full((3, 4), float(k)) * 1e-6 for k in range(3)]
    surface = Surface.from_layers(scans, x_scale=1e-6, y_scale=1e-6, z_type=DataType.FLOAT32)
    assert stacked(surface).shape == (3, 3, 4)
    assert surface.header.z.data_type is DataType.FLOAT32
    for index, layer in enumerate(surface.layers):
        np.testing.assert_array_equal(layer.z, scans[index])
    again = Surface.from_layers(surface.layers)  # layer objects bring their axes
    np.testing.assert_array_equal(stacked(again), stacked(surface))
    assert stacked(Surface.from_layers([np.zeros((2, 2))], x_scale=1.0, y_scale=1.0)).shape == (
        1,
        2,
        2,
    )


def test_a_profile_is_made_from_its_layers() -> None:
    profile = Profile.from_layers([np.zeros(4), np.ones(4)], x_scale=1e-6)
    assert stacked(profile).shape == (2, 4)
    again = Profile.from_layers(profile.layers)
    np.testing.assert_array_equal(stacked(again), stacked(profile))
    assert again.header == profile.header


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ([], "at least one layer"),
        ([np.zeros(3)], "2-D array"),
        ([np.zeros((2, 2)), np.zeros((2, 3))], "same shape"),
    ],
)
def test_the_layers_of_a_surface_have_to_fit_together(layers: list[object], message: str) -> None:
    with pytest.raises(X3pFormatError, match=message):
        Surface.from_layers(layers, x_scale=1.0, y_scale=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("layers", "message"),
    [
        ([], "at least one layer"),
        ([np.zeros((2, 2))], "1-D array"),
        ([np.zeros(2), np.zeros(3)], "same shape"),
    ],
)
def test_the_layers_of_a_profile_have_to_fit_together(layers: list[object], message: str) -> None:
    with pytest.raises(X3pFormatError, match=message):
        Profile.from_layers(layers, x_scale=1.0)  # type: ignore[arg-type]


def test_the_points_of_one_layer() -> None:
    surface = Surface.from_layers(
        [np.zeros((2, 3)), np.ones((2, 3))], x_scale=0.5, y_scale=2.0
    ).with_placement(Placement.translation(10.0, 0.0, 0.0))
    first, second = surface.layers
    assert first.points().shape == second.points().shape == (2, 3, 3)
    assert first.points()[..., 2].tolist() == [[0.0] * 3] * 2
    assert second.points()[..., 2].tolist() == [[1.0] * 3] * 2
    assert first.points()[0, :, 0].tolist() == [10.0, 10.5, 11.0]
    np.testing.assert_array_equal(surface.layers[-1].points(), second.points())
    np.testing.assert_array_equal(
        second.points(coordinate_system=x3pio.CoordinateSystem.VIEW)[0, :, 0], [0.0, 0.5, 1.0]
    )
    assert len(list(second.iter_points())) == 6
    surface = Surface.from_layers(surface.layers)  # a copy has read-only arrays
    surface.layers[1].z[0, 0] = np.nan
    assert len(list(surface.layers[1].iter_points(drop_invalid=True))) == 5
    with pytest.raises(IndexError):
        _ = surface.layers[2]


def test_the_points_of_a_profile_layer_and_of_a_point_cloud() -> None:
    profile = Profile.from_layers([np.zeros(3), np.ones(3)], x_scale=1.0)
    assert profile.layers[1].points().shape == (3, 3)
    assert profile.layers[1].points()[:, 2].tolist() == [1.0, 1.0, 1.0]
    cloud = PointCloud.from_points(np.arange(6.0).reshape(2, 3))
    assert len(cloud.layers) == 1
    assert cloud.layer is not None
    assert cloud.layer.points().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert cloud.layer.valid.tolist() == [True, True]


def test_a_layer_made_alone_takes_the_data_of_one_layer() -> None:
    with pytest.raises(ValueError, match="not 2"):
        ProfileLayer.from_array(np.zeros((2, 3)), x_scale=1.0)
    with pytest.raises(ValueError, match="not 3"):
        SurfaceLayer.from_array(np.zeros((3, 2, 2)), x_scale=1.0, y_scale=1.0)
    assert ProfileLayer.from_array(np.zeros(3), x_scale=1.0).z.shape == (3,)


def test_only_a_point_cloud_has_a_layer() -> None:
    assert not hasattr(Profile.from_array(np.zeros(3), x_scale=1.0), "layer")
    assert not hasattr(Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0), "layer")
    assert hasattr(PointCloud.from_points(np.zeros((1, 3))), "layer")


def test_a_layer_gives_its_points_without_the_file() -> None:
    placement = Placement.translation(1.0, 2.0, 3.0)
    surface = Surface.from_array(
        np.arange(6.0).reshape(2, 3),
        x_scale=0.5,
        y_scale=2.0,
        placement=placement,
        coordinate_system=CoordinateSystem.VIEW,
    )
    alone = SurfaceLayer.from_array(
        np.arange(6.0).reshape(2, 3),
        x_scale=0.5,
        y_scale=2.0,
        placement=placement,
        coordinate_system=CoordinateSystem.VIEW,
    )
    np.testing.assert_array_equal(alone.points(), surface.layers[0].points())
    assert alone.placement == placement
    line = ProfileLayer.from_array(np.zeros(4), x_scale=1.0)
    assert line.x_values.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert PointLayer.from_points(np.zeros((2, 3))).points().shape == (2, 3)


def test_the_layers_of_a_file_are_made_from_layers_of_other_files() -> None:
    first = Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0)
    second = Surface.from_array(np.ones((2, 2)), x_scale=1.0, y_scale=1.0)
    both = Surface.from_layers([first.layers[0], second.layers[0]])
    assert len(both.layers) == 2
    assert both.layers[1].z.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(TypeError, match="scales"):
        Surface.from_layers([first.layers[0]], x_scale=1.0)
    other = Surface.from_array(np.ones((2, 2)), x_scale=2.0, y_scale=1.0)
    with pytest.raises(X3pFormatError, match="axes and placement"):
        Surface.from_layers([first.layers[0], other.layers[0]])
    moved = first.with_placement(Placement.translation(1.0, 0.0, 0.0))
    with pytest.raises(X3pFormatError, match="axes and placement"):
        Surface.from_layers([first.layers[0], moved.layers[0]])
    with pytest.raises(X3pFormatError, match="either all as arrays or all as layers"):
        Surface.from_layers([first.layers[0], np.zeros((2, 2))])


def test_layers_with_coordinates_of_their_own_make_a_file() -> None:
    grid = _irregular_grid()
    irregular = Surface.from_points(np.stack([grid, grid + 1.0]))
    again = Surface.from_layers(irregular.layers, metadata=irregular.metadata)
    assert again == irregular
    assert again.layers[1].x is not None


def _wrong_options() -> list[Callable[[], X3pFile]]:
    one = Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
    return [
        lambda: Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0, **{"z_typ": 1}),  # type: ignore[arg-type]
        lambda: Profile.from_array(np.zeros(2), x_scale=1.0, **{"metdata": None}),  # type: ignore[arg-type]
        lambda: Surface.from_layers([np.zeros((1, 2))], x_scale=1.0, y_scale=1.0, **{"colour": 1}),  # type: ignore[arg-type]
        lambda: Surface.from_layers(one.layers, **{"z_scale": 1.0}),  # type: ignore[arg-type]
        lambda: Surface.from_points(np.zeros((1, 2, 3)), **{"z_type": DataType.INT16}),  # type: ignore[arg-type]
        lambda: PointCloud.from_points(np.zeros((2, 3)), **{"scalee": 1.0}),  # type: ignore[arg-type]
    ]


@pytest.mark.parametrize("make", _wrong_options())
def test_an_option_that_does_not_exist_is_refused(make: Callable[[], X3pFile]) -> None:
    with pytest.raises(TypeError, match="Unexpected option"):
        make()
