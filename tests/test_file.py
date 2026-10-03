# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import x3pio
from helpers import (
    SURFACE,
    absolute,
    array_file,
    build,
    edit_main,
    flat_points,
    rezip,
    stacked,
    unzip,
)
from x3pio import (
    Axis,
    AxisType,
    DataStorage,
    DataType,
    FeatureType,
    Header,
    Placement,
    PointCloud,
    Profile,
    Surface,
    SurfaceLayer,
    X3pFile,
    X3pFormatError,
)

ALL_STORAGES = list(DataStorage)
ALL_TYPES = list(DataType)


def _tolerance(data_type: DataType, scale: float) -> float:
    return {DataType.FLOAT64: 1e-18, DataType.FLOAT32: 1e-12}.get(data_type, scale / 2 + 1e-18)


def test_write_read_roundtrip_via_path(tmp_path: Path) -> None:
    path = tmp_path / "surface.x3p"
    x3pio.write(path, SURFACE, x_scale=1e-6, y_scale=2e-6)

    x3p = Surface.open(path)

    assert stacked(x3p)[0].shape == (2, 3)
    np.testing.assert_allclose(stacked(x3p)[0], SURFACE, rtol=1e-12, equal_nan=True)
    assert isinstance(x3p, Surface)
    assert x3p.feature_type is FeatureType.SURFACE
    assert x3p.header.x.increment == 1e-6
    assert x3p.header.y.increment == 2e-6


@pytest.mark.parametrize("storage", ALL_STORAGES)
@pytest.mark.parametrize("z_type", ALL_TYPES)
def test_write_read_roundtrip(storage: DataStorage, z_type: DataType) -> None:
    buffer = io.BytesIO()
    x3pio.write(buffer, SURFACE, x_scale=1e-6, y_scale=2e-6, z_type=z_type, storage=storage)

    x3p = x3pio.read(io.BytesIO(buffer.getvalue()))

    assert x3p.storage is storage
    assert x3p.header.z.data_type is z_type
    assert np.array_equal(np.isnan(stacked(x3p)), np.isnan(SURFACE[None]))
    scale = x3p.header.z.increment
    np.testing.assert_allclose(
        stacked(x3p), SURFACE[None], atol=_tolerance(z_type, scale), equal_nan=True
    )


def test_float_types_store_metres_directly() -> None:
    x3p = x3pio.read(io.BytesIO(build(z_type=DataType.FLOAT32)))
    assert x3p.header.z.increment == 1.0


def test_integer_types_pick_the_finest_increment_that_fits() -> None:
    x3p = x3pio.read(io.BytesIO(build(DataStorage.BINARY, z_type=DataType.INT16)))
    assert x3p.header.z.increment == pytest.approx(6e-6 / 32767)
    raw = x3p.raw_data()["z"]
    assert raw.dtype == np.dtype("<i2")
    assert raw.max() == 32767


def test_explicit_z_scale_is_used() -> None:
    x3p = x3pio.read(io.BytesIO(build(z_scale=1e-9, z_type=DataType.INT32)))
    assert x3p.header.z.increment == 1e-9
    assert x3p.raw_data()["z"][0] == 1000


def test_integer_overflow_is_rejected() -> None:
    with pytest.raises(X3pFormatError, match="out of range"):
        build(z_scale=1e-12, z_type=DataType.INT16)


def test_binary_file_layout() -> None:
    members = unzip(build(DataStorage.BINARY, z_type=DataType.FLOAT32, z_scale=1.0))
    assert sorted(members) == ["bindata/data.bin", "main.xml", "md5checksum.hex"]
    values = np.frombuffer(members["bindata/data.bin"], dtype="<f4")
    np.testing.assert_allclose(values, SURFACE.ravel().astype("<f4"), equal_nan=True)


def test_validity_file_is_only_written_for_integers_with_missing_points() -> None:
    members = unzip(build(DataStorage.BINARY, z_type=DataType.INT16))
    assert members["bindata/valid.bin"] == bytes([0b00111011])
    assert "bindata/valid.bin" not in unzip(build(DataStorage.BINARY, z_type=DataType.FLOAT64))
    full = np.ones((2, 3)) * 1e-6
    assert "bindata/valid.bin" not in unzip(build(DataStorage.BINARY, full, z_type=DataType.INT16))


def test_missing_point_of_an_integer_file_is_marked_by_the_validity_file_only() -> None:
    members = unzip(build(DataStorage.BINARY, z_type=DataType.INT16))
    raw = np.frombuffer(members["bindata/data.bin"], dtype="<i2")
    assert raw[2] == 0
    x3p = x3pio.X3pFile.loads(rezip(members))
    assert np.isnan(stacked(x3p)[0, 0, 2])
    assert stacked(x3p)[0, 0, 0] != 0


def test_xml_storage_writes_empty_datum_for_missing_points() -> None:
    text = unzip(build())["main.xml"].decode()
    assert text.count("<Datum />") == 1
    assert text.count("<Datum>") == 5


def test_xml_and_binary_storage_hold_the_same_data() -> None:
    xml = X3pFile.loads(build(DataStorage.XML))
    binary = X3pFile.loads(xml.dumps(storage=DataStorage.BINARY))
    assert xml == binary
    assert xml.storage is DataStorage.XML
    assert binary.storage is DataStorage.BINARY


def test_dumps_can_override_the_storage_and_is_deterministic() -> None:
    x3p = X3pFile.loads(build())
    as_binary = x3p.dumps(storage=DataStorage.BINARY)
    assert "bindata/data.bin" in unzip(as_binary)
    assert as_binary == x3p.dumps(storage=DataStorage.BINARY)
    assert X3pFile.loads(as_binary) == x3p
    with pytest.raises(TypeError, match="DataStorage"):
        x3p.dumps(storage="binary")  # type: ignore[arg-type]


def test_save_to_stream_and_path(tmp_path: Path) -> None:
    x3p = X3pFile.loads(build())
    buffer = io.BytesIO()
    x3p.save(buffer)
    path = tmp_path / "copy.x3p"
    x3p.save(path)
    assert path.read_bytes() == buffer.getvalue()


def test_layers_are_kept() -> None:
    stack = np.arange(24, dtype=float).reshape(2, 3, 4) * 1e-6
    buffer = io.BytesIO()
    x3pio.write(buffer, stack, x_scale=1e-6, y_scale=1e-6)
    x3p = Surface.open(io.BytesIO(buffer.getvalue()))
    assert stacked(x3p).shape == (2, 3, 4)
    assert np.array_equal(stacked(x3p), stack)
    assert x3p.is_regular_grid  # of every layer, they share their axes
    assert len(x3p.layers) == 2
    for index, layer in enumerate(x3p.layers):
        assert np.array_equal(layer.z, stack[index])
        assert layer.x is None


def test_profile() -> None:
    heights = np.array([1.0, 2.0, np.nan, 4.0]) * 1e-6
    buffer = io.BytesIO()
    x3pio.write(buffer, heights, x_scale=1e-6)
    x3p = x3pio.read(io.BytesIO(buffer.getvalue()))
    assert isinstance(x3p, Profile)
    assert x3p.feature_type is FeatureType.PROFILE
    assert stacked(x3p).shape == (1, 4)
    np.testing.assert_allclose(stacked(x3p)[0], heights, equal_nan=True)
    assert not hasattr(x3p, "surface")


def test_a_profile_of_several_layers() -> None:
    heights = np.arange(20.0).reshape(2, 10) * 1e-6
    x3p = Profile.from_array(heights, x_scale=1e-6)
    assert stacked(x3p).shape == (2, 10)
    assert len(x3p.layers) == 2
    assert [layer.z.shape for layer in x3p.layers] == [(10,), (10,)]
    again = x3pio.read(io.BytesIO(x3p.dumps()))
    assert isinstance(again, Profile)
    assert len(again.layers) == 2
    np.testing.assert_allclose(stacked(again), heights, rtol=1e-12)
    assert flat_points(again).shape == (20, 3)


def test_the_type_decides_what_a_file_can_be_asked() -> None:
    surface = X3pFile.loads(build())
    assert isinstance(surface, Surface)
    assert not hasattr(surface, "profile")


def test_axes_follow_the_standard() -> None:
    x3p = Surface.loads(build())
    assert x3p.layers[0].x_values.tolist() == pytest.approx([0.0, 1e-6, 2e-6])
    assert x3p.layers[0].y_values.tolist() == pytest.approx([0.0, 2e-6])
    assert x3p.is_regular_grid
    assert x3p.layers[0].valid.tolist() == [[True, True, False], [True, True, True]]


def test_y_increases_with_the_row_index() -> None:
    points = flat_points(X3pFile.loads(build()))
    assert points[3].tolist() == pytest.approx([0.0, 2e-6, 4e-6])


def test_a_list_has_no_axes_layers_or_grid() -> None:
    cloud = PointCloud.from_points(np.zeros((2, 3)))
    assert isinstance(cloud, PointCloud)
    for name in ("x_axis", "y_axis", "surface", "profile", "num_layers", "is_regular_grid"):
        assert not hasattr(cloud, name)


def _absolute_surface(x_type: DataType, y_type: DataType) -> Surface:
    shape = (1, 2, 3)
    x = np.array([[[0.0, 1.0, 2.5], [0.0, 1.5, 3.0]]]) * 1e-3
    y = np.array([[[0.0, 0.0, 0.1], [1.0, 1.0, 1.2]]]) * 1e-3
    z = np.arange(6.0).reshape(shape) * 1e-6
    z[0, 1, 1] = np.nan
    header = Header(
        x=Axis(AxisType.ABSOLUTE, x_type, 1e-6 if x_type.name.startswith("INT") else 1.0),
        y=Axis(AxisType.ABSOLUTE, y_type, 1e-6 if y_type.name.startswith("INT") else 1.0),
        z=Axis(AxisType.ABSOLUTE, DataType.FLOAT64, 1.0),
    )
    return Surface.from_stacked(header, z, x=x, y=y)


@pytest.mark.parametrize("storage", ALL_STORAGES)
@pytest.mark.parametrize(
    "x_type", [DataType.INT16, DataType.INT32, DataType.FLOAT32, DataType.FLOAT64]
)
def test_absolute_x_and_y_axes_roundtrip(storage: DataStorage, x_type: DataType) -> None:
    original = _absolute_surface(x_type, DataType.INT32)
    again = Surface.loads(original.dumps(storage=storage))
    assert stacked(again, "x") is not None
    assert stacked(again, "y") is not None
    assert stacked(again).shape == (1, 2, 3)
    np.testing.assert_allclose(stacked(again), stacked(original), equal_nan=True)
    np.testing.assert_allclose(stacked(again, "x")[0, 0], stacked(original, "x")[0, 0], atol=1e-6)  # type: ignore[index]
    assert not again.is_regular_grid
    with pytest.raises(ValueError, match="x_values"):
        _ = again.layers[0].x_values


def test_mixed_record_layout_is_packed_without_padding() -> None:
    original = _absolute_surface(DataType.INT16, DataType.FLOAT64)
    members = unzip(original.dumps(storage=DataStorage.BINARY))
    assert len(members["bindata/data.bin"]) == 6 * (2 + 8 + 8)
    assert original.raw_data().dtype.names == ("x", "y", "z")
    assert original.raw_data().dtype.itemsize == 18


def test_offset_and_rotation_roundtrip() -> None:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    placement = Placement(rotation, offset=[0.5, 0.0, -2.0])
    x3p = X3pFile.loads(build()).with_placement(placement)
    again = X3pFile.loads(x3p.dumps())
    assert again.header == x3p.header
    assert again.placement == placement


def test_equality_is_nan_aware_and_array_safe() -> None:
    blob = build()
    first = X3pFile.loads(blob)
    second = X3pFile.loads(blob)
    assert first == second
    stacked(second)[0, 0, 0] = 0.0
    assert first != second
    assert first != "file"
    third = X3pFile.loads(blob)
    third.metadata = None
    assert first != third
    fourth = X3pFile.loads(blob)
    fourth.extensions["http://a.b/c.xml"] = b"1"
    assert first != fourth


def test_equality_compares_absolute_axis_arrays() -> None:
    first = _absolute_surface(DataType.FLOAT64, DataType.FLOAT64)
    second = _absolute_surface(DataType.FLOAT64, DataType.FLOAT64)
    assert first == second
    assert stacked(second, "x") is not None
    absolute(second, "x")[0, 0, 0] = 5.0
    assert first != second
    assert first != Surface([SurfaceLayer(second.layers[0].z, header=second.layers[0].header)])


def test_with_z_type_returns_a_copy() -> None:
    x3p = X3pFile.loads(build())
    retyped = x3p.with_z_type(DataType.INT16)
    assert retyped.header.z.data_type is DataType.INT16
    assert x3p.header.z.data_type is DataType.FLOAT64
    again = X3pFile.loads(retyped.dumps())
    np.testing.assert_allclose(
        stacked(again), stacked(x3p), atol=retyped.header.z.increment, equal_nan=True
    )
    assert x3p.with_z_type(DataType.INT32, increment=1e-9).header.z.increment == 1e-9
    with pytest.raises(X3pFormatError, match="positive"):
        x3p.with_z_type(DataType.INT32, increment=0.0)


def test_from_array_needs_y_scale_for_rows() -> None:
    with pytest.raises(TypeError, match="y_scale"):
        array_file(np.zeros((2, 2)), x_scale=1.0)
    assert array_file(np.zeros(3), x_scale=2.0).header.y.increment == 2.0


@pytest.mark.parametrize("scale", [0.0, -1e-6])
def test_from_array_rejects_non_positive_scales(scale: float) -> None:
    with pytest.raises(X3pFormatError, match="positive"):
        array_file(np.zeros((2, 2)), x_scale=scale, y_scale=1.0)


def test_from_array_rejects_wrong_dimensions() -> None:
    with pytest.raises(X3pFormatError, match="1-D profile"):
        array_file(np.zeros((1, 1, 1, 1)), x_scale=1.0, y_scale=1.0)


def test_from_array_accepts_lists_and_infinity_marks_a_missing_point() -> None:
    x3p = array_file([[1.0, float("inf")]], x_scale=1.0, y_scale=1.0)
    again = X3pFile.loads(x3p.dumps())
    assert np.isnan(stacked(again)[0, 0, 1])


@pytest.mark.parametrize(
    ("file_type", "header", "data", "message"),
    [
        (PointCloud, Header(), np.zeros((1, 1, 1)), "must be 1-D"),
        (Profile, Header(), np.zeros((1, 2, 3)), "must be 1-D"),
        (Profile, Header(), np.zeros(3), "must be 1-D"),
        (Surface, Header(), np.zeros(3), "must be 2-D"),
        (Surface, Header(), np.zeros((2, 2)), "must be 2-D"),
        (Surface, Header(z=Axis(AxisType.INCREMENTAL)), np.zeros((1, 1, 1)), "must be absolute"),
        (
            Surface,
            Header(x=Axis(AxisType.ABSOLUTE)),
            np.zeros((1, 1, 1)),
            "absolute x axis needs",
        ),
        (Surface, Header(x=Axis(increment=0.0)), np.zeros((1, 1, 1)), "positive"),
    ],
)
def test_inconsistent_files_cannot_be_written(
    file_type: type[X3pFile], header: Header, data: np.ndarray, message: str
) -> None:
    with pytest.raises(X3pFormatError, match=message):
        file_type.from_stacked(header, data).dumps()


def test_incremental_axis_with_coordinates_cannot_be_written() -> None:
    x3p = Surface.from_stacked(Header(), np.zeros((1, 1, 1)), x=np.zeros((1, 1, 1)))
    with pytest.raises(X3pFormatError, match="incremental, but has coordinates"):
        x3p.dumps()


def test_absolute_axis_with_wrongly_shaped_coordinates_cannot_be_written() -> None:
    header = Header(x=Axis(AxisType.ABSOLUTE))
    with pytest.raises(X3pFormatError, match="shaped like data"):
        Surface.from_stacked(header, np.zeros((1, 2, 2)), x=np.zeros((1, 1, 1))).dumps()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_xml_writes_a_point_with_a_non_finite_coordinate_as_not_measured(value: float) -> None:
    original = _absolute_surface(DataType.FLOAT64, DataType.FLOAT64)
    assert stacked(original, "x") is not None
    absolute(original, "x")[0, 0, 0] = value
    stacked(original)[0, 0, 1] = value
    again = X3pFile.loads(original.dumps(storage=DataStorage.XML))
    assert np.isnan(stacked(again)[0, 0, 0])
    assert np.isnan(stacked(again)[0, 0, 1])
    assert np.isfinite(stacked(again)[0, 1, 0])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_non_finite_coordinate_of_an_integer_file_is_marked_in_the_validity_file(
    value: float,
) -> None:
    original = _absolute_surface(DataType.INT16, DataType.INT16)
    assert stacked(original, "x") is not None
    absolute(original, "x")[0, 0, 0] = value
    again = X3pFile.loads(original.dumps(storage=DataStorage.BINARY))
    assert np.isnan(stacked(again)[0, 0, 0])
    assert np.isnan(absolute(again, "x")[0, 0, 0]) if absolute(again, "x") is not None else False
    assert np.isfinite(stacked(again)[0, 1, 0])


def test_read_rejects_files_that_are_not_x3p() -> None:
    with pytest.raises(X3pFormatError, match="not a zip"):
        X3pFile.loads(b"nope")


def test_read_rejects_data_files_of_the_wrong_size() -> None:
    members = unzip(build(DataStorage.BINARY))
    members["bindata/data.bin"] = members["bindata/data.bin"][:-1]
    with pytest.raises(X3pFormatError, match="holds 47 bytes, but the dimensions and axes need 48"):
        X3pFile.loads(rezip(members))


def test_read_rejects_validity_files_of_the_wrong_size() -> None:
    members = unzip(build(DataStorage.BINARY, z_type=DataType.INT16))
    members["bindata/valid.bin"] += b"\x00"
    with pytest.raises(X3pFormatError, match="validity file"):
        X3pFile.loads(rezip(members))


@pytest.mark.parametrize("member", ["bindata/data.bin", "bindata/valid.bin"])
def test_read_rejects_missing_data_files(member: str) -> None:
    members = unzip(build(DataStorage.BINARY, z_type=DataType.INT16))
    del members[member]
    with pytest.raises(X3pFormatError, match="missing from the container"):
        X3pFile.loads(rezip(members))


def test_read_rejects_links_out_of_the_container() -> None:
    blob = edit_main(
        build(DataStorage.BINARY),
        lambda text: text.replace("bindata/data.bin", "../data.bin"),
    )
    with pytest.raises(X3pFormatError, match="does not point to a member"):
        X3pFile.loads(blob)


def test_read_rejects_integers_that_are_not_whole_in_xml() -> None:
    blob = build(z_type=DataType.INT16)
    blob = edit_main(blob, lambda text: text.replace("<Datum>", "<Datum>0.5", 1))
    with pytest.raises(X3pFormatError, match="whole numbers"):
        X3pFile.loads(blob)


def test_read_rejects_integers_out_of_range_in_xml() -> None:
    blob = build(z_type=DataType.INT16)
    blob = edit_main(blob, lambda text: text.replace("<Datum>", "<Datum>99999", 1))
    with pytest.raises(X3pFormatError, match="whole numbers"):
        X3pFile.loads(blob)


def test_larger_file_roundtrip() -> None:
    rng = np.random.default_rng(1)
    heights = rng.normal(0.0, 1e-6, (120, 200))
    heights[5, 7] = np.nan
    buffer = io.BytesIO()
    x3pio.write(buffer, heights, x_scale=1e-6, y_scale=1e-6)
    again = Surface.open(io.BytesIO(buffer.getvalue()))
    assert np.array_equal(stacked(again)[0], heights, equal_nan=True)


def test_file_can_be_read_from_an_open_stream() -> None:
    with io.BytesIO(build()) as stream:
        assert stacked(x3pio.read(stream)).shape == (1, 2, 3)


def test_validity_file_marks_the_whole_point_of_an_integer_axis() -> None:
    original = _absolute_surface(DataType.INT16, DataType.FLOAT64)
    members = unzip(original.dumps(storage=DataStorage.BINARY))
    # The z axis is a float, but the integer x axis cannot hold NaN either.
    assert "bindata/valid.bin" in members
    again = X3pFile.loads(original.dumps(storage=DataStorage.BINARY))
    assert stacked(again, "x") is not None
    assert stacked(again, "y") is not None
    assert np.isnan(absolute(again, "x")[0, 1, 1])
    assert np.isnan(absolute(again, "y")[0, 1, 1])
    assert np.isfinite(absolute(again, "x")[0, 1, 0])


def test_coordinates_of_a_missing_point_survive_in_float_files() -> None:
    original = _absolute_surface(DataType.FLOAT64, DataType.FLOAT64)
    assert "bindata/valid.bin" not in unzip(original.dumps(storage=DataStorage.BINARY))
    again = X3pFile.loads(original.dumps(storage=DataStorage.BINARY))
    assert stacked(again, "x") is not None
    assert absolute(again, "x")[0, 1, 1] == 1.5e-3
    assert np.isnan(stacked(again)[0, 1, 1])


def test_a_placement_that_is_not_finite_or_not_a_rotation_cannot_be_written() -> None:
    x3p = Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
    off = x3p.with_placement(Placement._unchecked(np.eye(3), np.array([np.nan, 0.0, 0.0])))
    with pytest.raises(X3pFormatError, match="finite"):
        off.dumps()
    skew = x3p.with_placement(Placement._unchecked(np.eye(3) * 2.0, np.zeros(3)))
    with pytest.raises(X3pFormatError, match="within"):
        skew.dumps()
