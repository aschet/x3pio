# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Changing the values of a file: the arrays of a layer, other layers, and writable copies."""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np
import pytest

from x3pio import (
    DataStorage,
    Layer,
    Metadata,
    PointCloud,
    PointLayer,
    Profile,
    ProfileLayer,
    Revision,
    Surface,
    SurfaceLayer,
    X3pFile,
    X3pFormatError,
)


def _surface(**options: object) -> Surface:
    heights = np.arange(6.0).reshape(2, 3) * 1e-6
    return Surface.from_array(heights, x_scale=1e-6, y_scale=2e-6, **options)  # type: ignore[arg-type]


def _plain() -> Surface:
    return Surface.from_array(np.arange(6.0).reshape(2, 3), x_scale=1.0, y_scale=1.0)


# --- the arrays of a layer ------------------------------------------------------------------------


def test_heights_can_be_set_one_by_one_and_in_bulk() -> None:
    surface = _plain()
    layer = surface.layers[0]
    layer.z[0, 1] = 7.0
    layer.z -= 1.0
    assert layer.z.tolist() == [[-1.0, 6.0, 1.0], [2.0, 3.0, 4.0]]
    assert surface.layers[0].z is layer.z  # the file's data, not a copy
    layer.z = np.full((2, 3), 2.0)
    assert layer.z.tolist() == [[2.0] * 3] * 2
    layer.z = 0.5  # a single number fills the layer
    assert layer.z.tolist() == [[0.5] * 3] * 2


def test_assigning_fills_the_array_of_the_file() -> None:
    surface = _plain()
    array = surface.layers[0].z
    surface.layers[0].z = np.ones((2, 3))
    assert surface.layers[0].z is array
    assert X3pFile.loads(surface.dumps()).layers[0].z.tolist() == [[1.0] * 3] * 2


def test_a_value_of_another_shape_is_refused() -> None:
    layer = _plain().layers[0]
    with pytest.raises(ValueError, match=r"shape \(2, 3\), not \(3,\)"):
        layer.z = np.zeros(3)
    assert layer.z[1, 2] == 5.0


def test_a_layer_that_shares_its_arrays_is_read_only() -> None:
    layer = _plain().with_metadata(comment="x").layers[0]
    with pytest.raises(ValueError, match="copy"):
        layer.z = 1.0
    with pytest.raises(ValueError, match="read-only"):
        layer.z -= 1.0
    with pytest.raises(ValueError, match="read-only"):
        layer.z[0, 0] = 1.0


def test_stored_coordinates_can_be_set_if_the_axis_is_absolute() -> None:
    cloud = PointCloud.from_points(np.zeros((3, 3)))
    cloud.layer.x = np.array([1.0, 2.0, 3.0])
    y = cloud.layer.y
    assert y is not None
    y += 4.0
    assert cloud.layer.points()[:, :2].tolist() == [[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]]


def test_an_incremental_axis_has_nothing_to_set() -> None:
    layer = _surface().layers[0]
    assert layer.x is None
    with pytest.raises(X3pFormatError, match="from_points"):
        layer.x = np.zeros((2, 3))
    with pytest.raises(X3pFormatError, match="from_points"):
        layer.y = 0.0


# --- other layers --------------------------------------------------------------------------------


def test_a_file_of_other_layers_keeps_what_the_file_holds() -> None:
    surface = _surface(
        metadata=Metadata(comment="kept"),
        storage=DataStorage.XML,
        revision=Revision.ISO5436_2000,
    )
    surface.extensions.add("http://a.b", "c.bin", b"1")
    cropped = SurfaceLayer.from_array(surface.layers[0].z[:, :2], x_scale=1e-6, y_scale=2e-6)
    smaller = surface.with_layers([cropped])
    assert smaller.layers[0].z.shape == (2, 2)
    assert smaller.layers[0] is not cropped  # the layers are copied
    assert smaller.metadata == surface.metadata
    assert smaller.metadata is not surface.metadata
    assert smaller.extensions == surface.extensions
    assert smaller.storage is DataStorage.XML
    assert smaller.revision is Revision.ISO5436_2000  # not that of the layer
    assert surface.layers[0].z.shape == (2, 3)
    again = X3pFile.loads(smaller.dumps())
    assert again.layers[0].z.shape == (2, 2)
    assert again.extensions == surface.extensions


def test_a_file_of_the_points_of_other_layers_brings_their_geometry() -> None:
    surface = _surface(metadata=Metadata(comment="kept"))
    points = surface.layers[0].points()
    points[..., 0] *= 2.0
    moved = surface.with_layers([SurfaceLayer.from_points(points)])
    assert not moved.is_regular_grid
    assert np.allclose(moved.layers[0].points(), points, rtol=0, atol=1e-18)
    assert moved.metadata == surface.metadata


def test_a_file_can_get_more_layers() -> None:
    profile = Profile.from_array(np.zeros(3), x_scale=1.0)
    layer = profile.layers[0]
    stack = profile.with_layers([layer, ProfileLayer.from_array(np.ones(3), x_scale=1.0)])
    assert [item.z.tolist() for item in stack.layers] == [[0.0] * 3, [1.0] * 3]
    assert len(profile.layers) == 1


@pytest.mark.parametrize(
    ("layers", "error", "message"),
    [
        (lambda layer: [], X3pFormatError, "at least one layer"),
        (lambda layer: [PointLayer.from_points(np.zeros((1, 3)))], TypeError, "SurfaceLayer"),
        (
            lambda layer: [layer, SurfaceLayer.from_array(layer.z, x_scale=2.0, y_scale=2.0)],
            X3pFormatError,
            "same axes",
        ),
    ],
)
def test_other_layers_have_to_fit_the_type_and_each_other(
    layers: Callable[[SurfaceLayer], list[Layer]], error: type[Exception], message: str
) -> None:
    surface = _surface()
    with pytest.raises(error, match=message):
        surface.with_layers(layers(surface.layers[0]))


def test_a_point_cloud_takes_one_layer() -> None:
    cloud = PointCloud.from_points(np.zeros((2, 3)))
    replaced = cloud.with_layers([PointLayer.from_points(np.ones((4, 3)))])
    assert replaced.layer.points().shape == (4, 3)


# --- layers into a new file ----------------------------------------------------------------------


def _stack() -> Surface:
    return Surface.from_array(
        np.arange(12.0).reshape(3, 2, 2),
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=Metadata(comment="stack"),
    )


def test_some_layers_of_a_file_make_a_new_file() -> None:
    stack = _stack()
    part = Surface.from_layers(stack.layers[1:], metadata=stack.metadata)
    assert [layer.z[0, 0] for layer in part.layers] == [4.0, 8.0]
    assert part.metadata == stack.metadata
    assert part.metadata is not stack.metadata
    part.layers[0].z[0, 0] = -1.0  # the new file has arrays of its own
    assert stack.layers[1].z[0, 0] == 4.0


def test_layers_of_a_new_file_may_come_from_several_files_and_in_any_order() -> None:
    stack = _stack()
    other = Surface.from_array(np.zeros((2, 2)), x_scale=1e-6, y_scale=1e-6)
    mixed = Surface.from_layers([*reversed(stack.layers), other.layers[0]])
    assert [layer.z[0, 0] for layer in mixed.layers] == [8.0, 4.0, 0.0, 0.0]


def test_layers_of_a_profile_make_a_new_profile_from_any_iterable() -> None:
    profile = Profile.from_array(np.arange(6.0).reshape(2, 3), x_scale=1.0)
    reversed_ = Profile.from_layers(reversed(profile.layers))
    assert [layer.z.tolist() for layer in reversed_.layers] == [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
    again = Profile.from_layers(layer for layer in profile.layers)
    assert again.layers == profile.layers


def test_layers_of_a_new_file_have_to_fit_together() -> None:
    stack = _stack()
    large = Surface.from_array(np.zeros((3, 3)), x_scale=1e-6, y_scale=1e-6)
    with pytest.raises(X3pFormatError, match="same shape"):
        Surface.from_layers([stack.layers[0], large.layers[0]])
    with pytest.raises(X3pFormatError, match="at least one layer"):
        Surface.from_layers(stack.layers[:0])


# --- copies ---------------------------------------------------------------------------------------


def test_copy_is_equal_independent_and_writable() -> None:
    original = _surface(metadata=Metadata(comment="x"))
    original.extensions["http://a.b/c.bin"] = b"1"
    shared = original.with_metadata(comment="y")
    duplicate = shared.copy()
    assert duplicate == shared
    assert duplicate.layers[0] is not shared.layers[0]
    duplicate.layers[0].z[0, 0] = 9.0
    assert shared.layers[0].z[0, 0] == 0.0
    assert duplicate.metadata is not shared.metadata
    assert duplicate.extensions is not shared.extensions
    assert duplicate.storage is shared.storage


def test_copy_of_a_file_with_absolute_axes_and_layers() -> None:
    cloud = PointCloud.from_points(np.arange(6.0).reshape(2, 3))
    duplicate = cloud.copy()
    duplicate.layer.x[0] = 5.0  # type: ignore[index]
    assert cloud.layer.x is not None
    assert cloud.layer.x[0] == 0.0
    stack = Surface.from_array(np.zeros((3, 2, 2)), x_scale=1.0, y_scale=1.0).copy()
    assert len(stack.layers) == 3


def test_the_copy_module_works_on_files_and_layers() -> None:
    surface = _surface()
    assert copy.deepcopy(surface) == surface
    assert copy.deepcopy(surface).layers[0].z is not surface.layers[0].z
    assert copy.copy(surface) == surface
    assert copy.deepcopy(surface.layers[0]) == surface.layers[0]
