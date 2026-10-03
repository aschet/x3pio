# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Where the stored points lie in space: the placement, the coordinate systems and their effect."""

from __future__ import annotations

import io

import numpy as np
import pytest

import x3pio
from helpers import absolute, flat_points, shaped_points, stacked, unzip
from x3pio import (
    CoordinateSystem,
    DataStorage,
    DataType,
    Placement,
    PointCloud,
    Profile,
    Surface,
    X3pFile,
    X3pFormatError,
    validate_rotation,
)

OFFSET = (0.01, 0.02, 0.003)
_C, _S = np.cos(0.3), np.sin(0.3)
ROTATION = np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]])
PLACEMENT = Placement(ROTATION, OFFSET)
POINTS = np.random.default_rng(3).uniform(0.0, 1e-3, (20, 3))
HEIGHTS = np.arange(6.0).reshape(2, 3) * 1e-6


def _round_trip(x3p: X3pFile) -> X3pFile:
    return X3pFile.loads(x3p.dumps())


# --- the placement ----------------------------------------------------------------------------


def test_the_default_placement_is_the_identity() -> None:
    placement = Placement()
    assert placement.is_identity
    assert not placement.has_rotation
    assert placement == Placement.identity()
    assert placement.rotation.tolist() == np.eye(3).tolist()
    assert placement.offset.tolist() == [0.0, 0.0, 0.0]


def test_a_placement_applies_the_rotation_and_then_the_offset() -> None:
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    placed = PLACEMENT.apply(points)
    np.testing.assert_allclose(placed[0], ROTATION @ points[0] + OFFSET, atol=1e-15)
    np.testing.assert_allclose(placed[1], OFFSET, atol=1e-15)
    np.testing.assert_allclose(PLACEMENT.apply(points[0]), placed[0], atol=1e-15)


def test_a_translation_only_moves() -> None:
    placement = Placement.translation(1.0, 2.0, 3.0)
    assert not placement.has_rotation
    assert placement.apply([[1.0, 1.0, 1.0]]).tolist() == [[2.0, 3.0, 4.0]]


def test_a_rotation_about_an_axis() -> None:
    quarter = Placement.from_axis_angle([0.0, 0.0, 5.0], np.pi / 2)  # the length does not matter
    np.testing.assert_allclose(quarter.apply([[1.0, 0.0, 0.0]]), [[0.0, 1.0, 0.0]], atol=1e-15)
    np.testing.assert_allclose(quarter.apply([[0.0, 0.0, 1.0]]), [[0.0, 0.0, 1.0]], atol=1e-15)
    about_x = Placement.from_axis_angle([1.0, 0.0, 0.0], np.pi / 2)
    np.testing.assert_allclose(about_x.apply([[0.0, 1.0, 0.0]]), [[0.0, 0.0, 1.0]], atol=1e-15)
    validate_rotation(about_x.rotation)


@pytest.mark.parametrize("axis", [(0.0, 0.0, 0.0), (1.0, 2.0), (np.nan, 0.0, 1.0)])
def test_an_axis_must_be_three_finite_numbers_that_are_not_all_zero(axis: object) -> None:
    with pytest.raises(X3pFormatError, match="axis"):
        Placement.from_axis_angle(axis, 1.0)  # type: ignore[arg-type]


def test_the_inverse_undoes_a_placement() -> None:
    points = np.random.default_rng(0).uniform(-1e-3, 1e-3, (30, 3))
    placed = PLACEMENT.apply(points)
    np.testing.assert_allclose(PLACEMENT.inverse().apply(placed), points, atol=1e-15)
    assert (PLACEMENT @ PLACEMENT.inverse()).apply(points).shape == points.shape
    np.testing.assert_allclose((PLACEMENT @ PLACEMENT.inverse()).apply(points), points, atol=1e-15)


def test_placements_compose_like_matrices() -> None:
    a = Placement(ROTATION, OFFSET)
    b = Placement.translation(1.0, 0.0, 0.0)
    points = np.array([[0.1, 0.2, 0.3]])
    np.testing.assert_allclose((a @ b).apply(points), a.apply(b.apply(points)), atol=1e-15)
    assert (a @ Placement()) == a
    assert (Placement() @ a) == a
    with pytest.raises(TypeError):
        _ = a @ 3  # type: ignore[operator]


def test_a_point_that_is_not_measured_is_unknown_in_every_coordinate_only_with_a_rotation() -> None:
    points = np.array([[1.0, 2.0, np.nan]])
    with np.errstate(all="raise"):
        assert np.isnan(PLACEMENT.apply(points)).all()
        known = Placement.translation(1.0, 1.0, 1.0).apply(points)
    assert known[0, :2].tolist() == [2.0, 3.0]
    assert np.isnan(known[0, 2])


def test_a_placement_is_immutable_and_copies_what_it_is_given() -> None:
    rotation = ROTATION.copy()
    offset = np.array(OFFSET)
    placement = Placement(rotation, offset)
    rotation[0, 0] = 5.0
    offset[0] = 5.0
    assert placement.rotation[0, 0] == ROTATION[0, 0]
    assert placement.offset[0] == OFFSET[0]
    with pytest.raises(ValueError, match="read-only"):
        placement.rotation[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        placement.offset[0] = 1.0
    with pytest.raises(AttributeError):
        placement.offset = np.zeros(3)  # type: ignore[misc]


def test_a_placement_has_a_value_and_a_representation() -> None:
    assert Placement(ROTATION, OFFSET) == PLACEMENT
    assert Placement(ROTATION, OFFSET) != Placement(ROTATION, (0.0, 0.0, 0.0))
    assert PLACEMENT != "placement"
    assert repr(Placement.translation(1.0, 2.0, 3.0)).endswith("offset=[1.0, 2.0, 3.0])")
    with pytest.raises(TypeError):
        hash(PLACEMENT)


def test_a_rotation_may_be_a_nested_list() -> None:
    assert Placement(ROTATION.tolist()).rotation.tolist() == ROTATION.tolist()


@pytest.mark.parametrize(
    "offset", [(1.0, 2.0), (1.0, 2.0, 3.0, 4.0), (0.0, 0.0, np.nan), (np.inf, 0.0, 0.0), 1.0]
)
def test_an_offset_must_be_three_finite_numbers(offset: object) -> None:
    with pytest.raises(X3pFormatError, match="offset"):
        Placement(offset=offset)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "rotation",
    [np.eye(2), np.ones((3, 3)) * 0.5, np.diag([1.0, 1.0, -1.0]), np.full((3, 3), np.nan)],
    ids=["shape", "not orthonormal", "mirror", "nan"],
)
def test_a_rotation_must_be_a_rotation(rotation: np.ndarray) -> None:
    with pytest.raises(X3pFormatError, match="otation"):
        Placement(rotation)


def test_the_rotation_check_accepts_rotations_and_names_the_mistake() -> None:
    validate_rotation(np.eye(3))
    validate_rotation(ROTATION)
    with pytest.raises(X3pFormatError, match=r"within \[-1, 1\]"):
        validate_rotation(np.eye(3) * 2)
    with pytest.raises(X3pFormatError, match="3x3"):
        validate_rotation(np.eye(2))


# --- point clouds -----------------------------------------------------------------------------


@pytest.mark.parametrize("storage", list(DataStorage))
def test_global_points_come_back_as_given(storage: DataStorage) -> None:
    x3p = PointCloud.from_points(
        POINTS, placement=PLACEMENT, coordinate_system=CoordinateSystem.GLOBAL, storage=storage
    )
    again = _round_trip(x3p)
    np.testing.assert_allclose(flat_points(again), POINTS, atol=1e-15)
    np.testing.assert_allclose(again.placement.rotation, ROTATION)
    assert again.placement.offset.tolist() == list(OFFSET)


def test_global_points_are_stored_in_the_view_system() -> None:
    x3p = PointCloud.from_points(
        POINTS, placement=PLACEMENT, coordinate_system=CoordinateSystem.GLOBAL
    )
    assert stacked(x3p, "x") is not None
    assert stacked(x3p, "y") is not None
    view = np.column_stack([absolute(x3p, "x"), absolute(x3p, "y"), stacked(x3p)])
    np.testing.assert_allclose(view, (POINTS - OFFSET) @ ROTATION, atol=1e-15)
    np.testing.assert_allclose(
        flat_points(x3p, coordinate_system=CoordinateSystem.VIEW), view, atol=1e-15
    )


def test_view_points_are_stored_untransformed() -> None:
    x3p = PointCloud.from_points(
        POINTS, placement=PLACEMENT, coordinate_system=CoordinateSystem.VIEW
    )
    assert stacked(x3p, "x") is not None
    assert stacked(x3p, "y") is not None
    np.testing.assert_array_equal(
        np.column_stack([absolute(x3p, "x"), absolute(x3p, "y"), stacked(x3p)]), POINTS
    )
    np.testing.assert_allclose(
        flat_points(x3p, coordinate_system=CoordinateSystem.VIEW), POINTS, atol=1e-15
    )
    np.testing.assert_allclose(flat_points(x3p), POINTS @ ROTATION.T + OFFSET, atol=1e-15)


def test_an_offset_alone_is_enough() -> None:
    x3p = PointCloud.from_points(
        POINTS, placement=Placement(offset=OFFSET), coordinate_system=CoordinateSystem.GLOBAL
    )
    assert not x3p.placement.has_rotation
    np.testing.assert_allclose(flat_points(_round_trip(x3p)), POINTS, atol=1e-15)


def test_a_rotation_alone_is_enough() -> None:
    x3p = PointCloud.from_points(
        POINTS, placement=Placement(ROTATION), coordinate_system=CoordinateSystem.GLOBAL
    )
    assert x3p.placement.offset.tolist() == [0.0, 0.0, 0.0]
    np.testing.assert_allclose(flat_points(_round_trip(x3p)), POINTS, atol=1e-15)


def test_a_file_that_is_read_is_written_again_the_same_way() -> None:
    original = PointCloud.from_points(
        POINTS,
        placement=PLACEMENT,
        coordinate_system=CoordinateSystem.GLOBAL,
        data_type=DataType.INT32,
    )
    again = PointCloud.from_points(
        flat_points(original),
        placement=original.placement,
        coordinate_system=CoordinateSystem.GLOBAL,
    )
    np.testing.assert_allclose(flat_points(again), flat_points(original), atol=1e-9)


def test_write_points_passes_the_placement() -> None:
    buffer = io.BytesIO()
    x3pio.write_points(
        buffer, POINTS, placement=PLACEMENT, coordinate_system=CoordinateSystem.GLOBAL
    )
    again = x3pio.read(io.BytesIO(buffer.getvalue()))
    np.testing.assert_allclose(flat_points(again), POINTS, atol=1e-15)


# --- surfaces and profiles --------------------------------------------------------------------


def _grid(**placement: object) -> Surface:
    return Surface.from_array(HEIGHTS, x_scale=1e-6, y_scale=1e-6, **placement)  # type: ignore[arg-type]


def test_view_heights_are_stored_untransformed() -> None:
    x3p = _grid(placement=PLACEMENT, coordinate_system=CoordinateSystem.VIEW)
    np.testing.assert_array_equal(stacked(x3p)[0], HEIGHTS)
    again = _round_trip(x3p)
    np.testing.assert_array_equal(stacked(again)[0], HEIGHTS)
    assert again.placement.offset[2] == OFFSET[2]
    assert again.placement.has_rotation
    np.testing.assert_allclose(flat_points(again)[0], OFFSET, atol=1e-15)


def test_global_heights_without_rotation_lose_the_z_offset() -> None:
    x3p = _grid(placement=Placement(offset=OFFSET), coordinate_system=CoordinateSystem.GLOBAL)
    np.testing.assert_allclose(stacked(x3p)[0], HEIGHTS - OFFSET[2])
    np.testing.assert_allclose(stacked(_round_trip(x3p).globalized())[0], HEIGHTS, atol=1e-15)


def test_global_heights_with_a_rotation_are_refused() -> None:
    with pytest.raises(X3pFormatError, match="regular grid"):
        _grid(placement=PLACEMENT, coordinate_system=CoordinateSystem.GLOBAL)


def test_a_profile_takes_the_placement_too() -> None:
    x3p = Profile.from_array(
        HEIGHTS[0],
        x_scale=1e-6,
        placement=Placement(offset=OFFSET),
        coordinate_system=CoordinateSystem.VIEW,
    )
    assert x3p.placement.offset[0] == OFFSET[0]
    assert _round_trip(x3p).placement.offset[2] == OFFSET[2]


def test_write_passes_the_placement() -> None:
    buffer = io.BytesIO()
    x3pio.write(
        buffer,
        HEIGHTS,
        x_scale=1e-6,
        y_scale=1e-6,
        placement=PLACEMENT,
        coordinate_system=CoordinateSystem.VIEW,
    )
    again = x3pio.read(io.BytesIO(buffer.getvalue()))
    assert again.placement.offset[1] == OFFSET[1]
    assert again.placement.has_rotation


def test_the_global_heights_survive_an_integer_type() -> None:
    x3p = _grid(
        placement=Placement(offset=OFFSET),
        coordinate_system=CoordinateSystem.GLOBAL,
        z_type=DataType.INT32,
    )
    np.testing.assert_allclose(stacked(_round_trip(x3p).globalized())[0], HEIGHTS, atol=1e-11)


def test_an_identity_rotation_is_not_written_and_a_real_one_is() -> None:
    plain = unzip(_grid().dumps())["main.xml"].decode()
    assert "<Rotation>" not in plain
    turned = unzip(_grid(placement=PLACEMENT, coordinate_system=CoordinateSystem.VIEW).dumps())[
        "main.xml"
    ].decode()
    assert "<Rotation>" in turned
    assert "<Offset>0.01</Offset>" in turned


# --- using a placement ------------------------------------------------------------------------


def test_the_two_coordinate_systems_give_the_points_as_stored_and_as_placed() -> None:
    x3p = _grid(placement=PLACEMENT, coordinate_system=CoordinateSystem.VIEW)
    view = flat_points(x3p, coordinate_system=CoordinateSystem.VIEW)
    np.testing.assert_allclose(
        flat_points(x3p, coordinate_system=CoordinateSystem.GLOBAL), PLACEMENT.apply(view)
    )
    np.testing.assert_allclose(flat_points(x3p), PLACEMENT.apply(view))
    assert [
        point.z for point in x3p.layers[0].iter_points(coordinate_system=CoordinateSystem.VIEW)
    ] == view[:, 2].tolist()


def test_the_points_are_shaped_like_the_file() -> None:
    surface = Surface.from_array(np.zeros((2, 3, 4)), x_scale=1.0, y_scale=1.0)
    profile = Profile.from_array(np.zeros((2, 5)), x_scale=1.0)
    cloud = PointCloud.from_points(POINTS)
    assert shaped_points(surface).shape == (2, 3, 4, 3)
    assert shaped_points(profile).shape == (2, 5, 3)
    assert shaped_points(cloud).shape == (20, 3)
    np.testing.assert_array_equal(shaped_points(surface).reshape(-1, 3), flat_points(surface))


def test_a_placement_can_be_replaced_without_touching_the_stored_data() -> None:
    x3p = _grid()
    moved = x3p.with_placement(PLACEMENT)
    assert x3p.placement.is_identity
    assert moved.placement == PLACEMENT
    np.testing.assert_array_equal(stacked(moved), stacked(x3p))
    np.testing.assert_allclose(flat_points(moved), PLACEMENT.apply(flat_points(x3p)), atol=1e-15)


def test_transforming_moves_the_points_relative_to_where_they_are() -> None:
    x3p = _grid(placement=PLACEMENT, coordinate_system=CoordinateSystem.VIEW)
    up = Placement.translation(0.0, 0.0, 1.0)
    moved = x3p.transformed(up)
    np.testing.assert_allclose(flat_points(moved), up.apply(flat_points(x3p)), atol=1e-15)
    np.testing.assert_array_equal(stacked(moved), stacked(x3p))
    assert moved.placement == up @ PLACEMENT
    assert x3p.placement == PLACEMENT  # the file that is transformed is as it was


def test_the_global_copy_has_the_global_points_stored() -> None:
    x3p = _grid(placement=PLACEMENT, coordinate_system=CoordinateSystem.VIEW)
    baked = x3p.globalized()
    assert baked.placement.is_identity
    np.testing.assert_allclose(
        flat_points(baked, coordinate_system=CoordinateSystem.VIEW), flat_points(x3p), atol=1e-15
    )
    np.testing.assert_allclose(flat_points(baked), flat_points(x3p), atol=1e-15)
    assert not baked.is_regular_grid


def test_a_surface_without_a_rotation_stays_a_regular_grid_in_the_global_copy() -> None:
    x3p = _grid(placement=Placement(offset=OFFSET), coordinate_system=CoordinateSystem.VIEW)
    baked = x3p.globalized()
    assert baked.is_regular_grid
    assert baked.placement.offset.tolist() == [OFFSET[0], OFFSET[1], 0.0]  # the first point stays
    np.testing.assert_allclose(flat_points(baked), flat_points(x3p), atol=1e-15)


def test_centering_keeps_the_global_points_and_uses_a_symmetric_range() -> None:
    far = POINTS + np.array([100.0, 200.0, 300.0])
    cloud = PointCloud.from_points(far)
    centered = cloud.centered()
    assert stacked(centered, "x") is not None
    assert abs(float(absolute(centered, "x").min()) + float(absolute(centered, "x").max())) < 1e-12
    assert abs(float(stacked(centered).min()) + float(stacked(centered).max())) < 1e-12
    np.testing.assert_allclose(flat_points(centered), flat_points(cloud), atol=1e-12)
    np.testing.assert_allclose(flat_points(_round_trip(centered)), far, atol=1e-12)
    assert cloud.placement.is_identity  # the file that is centred is as it was


def test_centering_lets_an_integer_type_use_its_range() -> None:
    heights = 1.0 + np.arange(6.0).reshape(2, 3) * 1e-6  # a level far from zero
    surface = Surface.from_array(heights, x_scale=1e-3, y_scale=1e-3)
    expected = flat_points(surface)[:, 2]
    plain = flat_points(_round_trip(surface.with_z_type(DataType.INT16)))[:, 2]
    better = flat_points(_round_trip(surface.centered().with_z_type(DataType.INT16)))[:, 2]
    assert np.abs(plain - expected).max() > 1e-6  # the range of the type is spent on the level
    assert np.abs(better - expected).max() < 1e-9


def test_centering_a_surface_centres_the_heights_only() -> None:
    surface = Surface.from_array(HEIGHTS + 1.0, x_scale=1e-3, y_scale=1e-3)
    centered = surface.centered()
    assert centered.is_regular_grid
    assert abs(float(np.nanmin(stacked(centered))) + float(np.nanmax(stacked(centered)))) < 1e-12
    assert centered.placement.offset[:2].tolist() == [0.0, 0.0]
    np.testing.assert_allclose(flat_points(centered), flat_points(surface), atol=1e-12)
    np.testing.assert_allclose(flat_points(_round_trip(centered)), flat_points(surface), atol=1e-12)


# --- mistakes ---------------------------------------------------------------------------------


def test_a_placement_needs_the_system_of_the_coordinates() -> None:
    with pytest.raises(TypeError, match="coordinate_system"):
        PointCloud.from_points(POINTS, placement=PLACEMENT)
    with pytest.raises(TypeError, match="coordinate_system"):
        _grid(placement=PLACEMENT)
    with pytest.raises(TypeError, match="coordinate_system"):
        x3pio.write(io.BytesIO(), HEIGHTS, x_scale=1.0, y_scale=1.0, placement=PLACEMENT)
    with pytest.raises(TypeError, match="coordinate_system"):
        x3pio.write_points(io.BytesIO(), POINTS, placement=PLACEMENT)


@pytest.mark.parametrize(
    "coordinate_system", [None, CoordinateSystem.GLOBAL, CoordinateSystem.VIEW]
)
def test_without_a_placement_the_system_does_not_matter(
    coordinate_system: CoordinateSystem | None,
) -> None:
    options = {} if coordinate_system is None else {"coordinate_system": coordinate_system}
    x3p = PointCloud.from_points(POINTS, **options)  # type: ignore[arg-type]
    np.testing.assert_array_equal(flat_points(x3p), POINTS)
    assert x3p.placement.is_identity
    assert _grid(**options).placement.is_identity


def test_the_coordinate_system_is_an_enum_and_no_text() -> None:
    assert {member.name for member in CoordinateSystem} == {"VIEW", "GLOBAL"}
    assert len(CoordinateSystem) == 2


# --- the increments that centring chooses ---------------------------------------------------------

FAR = POINTS + np.array([100.0, 50.0, 10.0])


def _written_error(x3p: X3pFile) -> float:
    return float(np.abs(flat_points(PointCloud.loads(x3p.dumps())) - FAR).max())


@pytest.mark.parametrize("data_type", [DataType.INT16, DataType.INT32])
def test_centring_gives_integer_axes_the_increment_of_the_centred_range(
    data_type: DataType,
) -> None:
    plain = PointCloud.from_points(FAR, data_type=data_type)
    centered = plain.centered()
    np.testing.assert_allclose(flat_points(centered), FAR, atol=1e-12)
    for name in ("x", "y", "z"):
        assert getattr(centered.header, name).increment < getattr(plain.header, name).increment
    assert _written_error(centered) < _written_error(plain) / 1e3


def test_centring_replaces_an_increment_that_was_given_for_an_integer_axis() -> None:
    plain = PointCloud.from_points(FAR, data_type=DataType.INT32, scale=1e-6)
    assert plain.header.x.increment == 1e-6
    assert plain.centered().header.x.increment != 1e-6


def test_centring_leaves_the_increments_of_float_axes_and_incremental_axes() -> None:
    cloud = PointCloud.from_points(FAR).centered()
    assert [axis.increment for axis in (cloud.header.x, cloud.header.y, cloud.header.z)] == [
        1.0
    ] * 3
    surface = Surface.from_array(
        FAR[:, 2].reshape(4, 5), x_scale=1e-6, y_scale=2e-6, z_type=DataType.INT16
    )
    centered = surface.centered()
    assert centered.header.x == surface.header.x
    assert centered.header.y == surface.header.y
    assert centered.header.z.increment < surface.header.z.increment
    np.testing.assert_allclose(flat_points(centered), flat_points(surface), atol=1e-12)
