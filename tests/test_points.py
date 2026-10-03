# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from dataclasses import replace

import numpy as np
import pytest

import x3pio
from helpers import absolute, flat_points, stacked
from x3pio import (
    Axis,
    AxisType,
    CoordinateSystem,
    DataStorage,
    DataType,
    FeatureType,
    Header,
    Placement,
    Point,
    PointCloud,
    Surface,
    X3pFile,
    X3pFormatError,
)

# Largest error a coordinate of up to 1e-3 can have after storage, per data type.
_TOLERANCE = {
    DataType.INT16: 1e-3 / 32767,
    DataType.INT32: 1e-3 / 2147483647,
    DataType.FLOAT32: 1e-3 * 2**-23,
    DataType.FLOAT64: 0.0,
}
ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _stack() -> Surface:
    # The 3 x 3 x 2 matrix of the example of the standard for the position index, z numbered in
    # storage order.
    data = np.arange(18, dtype=float).reshape(2, 3, 3)
    header = Header(x=Axis(increment=1.0), y=Axis(increment=10.0))
    return Surface.from_stacked(header, data)


def test_points_come_in_storage_order() -> None:
    points = flat_points(_stack())
    assert points.shape == (18, 3)
    assert points[:, 2].tolist() == list(range(18))
    # u is the fastest index, then v, then w.
    assert points[:3, 0].tolist() == [0.0, 1.0, 2.0]
    assert points[:3, 1].tolist() == [0.0, 0.0, 0.0]
    assert points[3, :2].tolist() == [0.0, 10.0]
    assert points[9, :2].tolist() == [0.0, 0.0]


def test_iter_points_matches_the_points_of_the_layers() -> None:
    x3p = _stack()
    points = [point for layer in x3p.layers for point in layer.iter_points()]
    assert points[4] == Point(1.0, 10.0, 4.0)
    assert np.array_equal(np.array(points), flat_points(x3p))


def test_global_coordinates_apply_rotation_and_offset() -> None:
    header = Header(
        x=Axis(increment=2.0),
        y=Axis(increment=3.0),
        z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
    )
    placement = Placement(ROTATION, offset=[10.0, 20.0, 30.0])
    x3p = Surface.from_stacked(header, np.array([[[5.0, 6.0]]]), placement=placement)
    view = flat_points(x3p, coordinate_system=CoordinateSystem.VIEW)
    assert view.tolist() == [[0.0, 0.0, 5.0], [2.0, 0.0, 6.0]]
    # Q = R * P + T, with P the view coordinates (the formula of Amd 1:2020).
    expected = view @ ROTATION.T + np.array([10.0, 20.0, 30.0])
    assert np.allclose(flat_points(x3p), expected)
    assert flat_points(x3p)[1].tolist() == [10.0, 22.0, 36.0]


def test_offset_without_rotation() -> None:
    placement = Placement.translation(0.0, 0.0, -0.5)
    surface = Surface.from_stacked(Header(), np.array([[[1.0]]]), placement=placement)
    assert flat_points(surface).tolist() == [[0.0, 0.0, 0.5]]


def test_identity_rotation_keeps_unmeasured_points_known() -> None:
    header = Header(x=Axis(increment=2.0))
    points = flat_points(
        Surface.from_stacked(header, np.array([[[1.0, np.nan]]]), placement=Placement())
    )
    assert points[1].tolist()[:2] == [2.0, 0.0]
    assert np.isnan(points[1, 2])


def test_rotation_makes_unmeasured_points_unknown() -> None:
    surface = Surface.from_stacked(
        Header(), np.array([[[1.0, np.nan]]]), placement=Placement(ROTATION)
    )
    points = flat_points(surface)
    assert np.isnan(points[1]).all()


def test_drop_invalid_removes_unmeasured_points() -> None:
    x3p = Surface.from_stacked(Header(), np.array([[[1.0, np.nan, 3.0]]]))
    assert flat_points(x3p).shape == (3, 3)
    assert flat_points(x3p, drop_invalid=True).shape == (2, 3)
    assert len(list(x3p.layers[0].iter_points(drop_invalid=True))) == 2


@pytest.mark.parametrize("storage", list(DataStorage))
@pytest.mark.parametrize("data_type", list(DataType))
def test_point_cloud_roundtrip(storage: DataStorage, data_type: DataType) -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(-1e-3, 1e-3, (50, 3))
    buffer = io.BytesIO()
    x3pio.write_points(buffer, points, data_type=data_type, storage=storage)

    x3p = x3pio.read(io.BytesIO(buffer.getvalue()))

    assert x3p.feature_type is FeatureType.POINT_CLOUD
    assert stacked(x3p).shape == (50,)
    assert isinstance(x3p, PointCloud)
    np.testing.assert_allclose(flat_points(x3p), points, atol=_TOLERANCE[data_type], rtol=0)


def test_point_cloud_keeps_the_order_of_the_points() -> None:
    points = np.array([[3.0, 1.0, 2.0], [-1.0, 5.0, 0.5], [0.0, 0.0, 0.0]])
    again = X3pFile.loads(PointCloud.from_points(points).dumps())
    assert np.array_equal(flat_points(again), points)
    assert stacked(again, "x") is not None
    assert absolute(again, "x").tolist() == [3.0, -1.0, 0.0]


def test_from_points_with_explicit_scale() -> None:
    x3p = PointCloud.from_points(
        np.array([[1e-6, 2e-6, 3e-6]]), data_type=DataType.INT32, scale=1e-9
    )
    assert x3p.header.x.increment == 1e-9
    assert x3p.raw_data()["z"].tolist() == [3000]


def test_from_points_picks_a_scale_for_every_axis() -> None:
    points = np.array([[1.0, 100.0, 0.01], [-2.0, 50.0, 0.005]])
    x3p = PointCloud.from_points(points, data_type=DataType.INT16)
    assert x3p.header.x.increment == pytest.approx(2.0 / 32767)
    assert x3p.header.y.increment == pytest.approx(100.0 / 32767)
    assert x3p.header.z.increment == pytest.approx(0.01 / 32767)


@pytest.mark.parametrize(
    ("points", "message"),
    [
        (np.zeros((3, 2)), r"\(N, 3\)"),
        (np.zeros(3), r"\(N, 3\)"),
        (np.array([[0.0, 0.0, np.nan]]), "not measured"),
        (np.array([[np.inf, 0.0, 1.0]]), "not measured"),
    ],
)
def test_from_points_rejects_bad_input(points: np.ndarray, message: str) -> None:
    with pytest.raises(X3pFormatError, match=message):
        PointCloud.from_points(points)


def test_point_cloud_cannot_hold_missing_points() -> None:
    cloud = PointCloud.from_points(np.zeros((2, 3)))
    stacked(cloud)[0] = np.nan
    with pytest.raises(X3pFormatError, match="not measured"):
        cloud.dumps()


def test_point_cloud_needs_absolute_axes_to_be_written() -> None:
    cloud = PointCloud.from_points(np.zeros((2, 3)))
    header = replace(cloud.header, x=Axis(AxisType.INCREMENTAL))
    incremental = PointCloud.from_stacked(header, cloud.layer.z, y=cloud.layer.y)
    with pytest.raises(X3pFormatError, match="must be absolute"):
        incremental.dumps()


def test_list_with_an_incremental_axis_uses_the_index() -> None:
    header = Header(x=Axis(increment=0.5), y=Axis(AxisType.ABSOLUTE))
    cloud = PointCloud.from_stacked(header, np.array([5.0, 6.0, 7.0]), y=np.array([1.0, 2.0, 3.0]))
    assert flat_points(cloud).tolist() == [[0.0, 1.0, 5.0], [0.5, 2.0, 6.0], [1.0, 3.0, 7.0]]


def test_points_of_a_surface_with_absolute_axes() -> None:
    x = np.array([[[0.0, 1.0], [0.5, 1.5]]])
    y = np.array([[[0.0, 0.1], [1.0, 1.1]]])
    header = Header(x=Axis(AxisType.ABSOLUTE), y=Axis(AxisType.ABSOLUTE))
    surface = Surface.from_stacked(header, np.arange(4.0).reshape(1, 2, 2), x=x, y=y)
    assert flat_points(surface).tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.1, 1.0],
        [0.5, 1.0, 2.0],
        [1.5, 1.1, 3.0],
    ]


def test_absolute_axis_without_coordinates_fails_to_give_points() -> None:
    header = Header(x=Axis(AxisType.ABSOLUTE))
    with pytest.raises(X3pFormatError, match="needs its coordinates"):
        flat_points(Surface.from_stacked(header, np.zeros((1, 1, 1))))
