# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The synthetic sample files in ``samples/``: they are current, valid and read as intended."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import x3pio
from helpers import SCHEMAS, flat_points, stacked, unzip
from x3pio import (
    DataStorage,
    DataType,
    PointCloud,
    Profile,
    Revision,
    Surface,
    X3pFile,
    get_data_type,
)
from x3pio.cli import main

FIRST = Revision.ISO5436_2000
AMENDED = Revision.ISO25178_72_2017_DAM1

#: The type and the number of layers of each sample.
KINDS: dict[str, tuple[type[X3pFile], int]] = {
    "profile": (Profile, 1),
    "profile_layers": (Profile, 3),
    "surface": (Surface, 1),
    "surface_layers": (Surface, 3),
    "surface_irregular": (Surface, 1),
    "surface_irregular_layers": (Surface, 3),
    "surface_extensions": (Surface, 1),
    "surface_int32": (Surface, 1),
    "surface_placed": (Surface, 1),
    "surface_absolute_placed": (Surface, 1),
    "profile_int16": (Profile, 1),
    "profile_int32": (Profile, 1),
    "point_cloud": (PointCloud, 1),
    "point_cloud_int16": (PointCloud, 1),
    "point_cloud_int32": (PointCloud, 1),
    "point_cloud_float64": (PointCloud, 1),
}
#: The data type of the heights, and of all axes of a point cloud.
DATA_TYPES = {
    "profile": DataType.FLOAT64,
    "profile_layers": DataType.FLOAT32,
    "profile_int16": DataType.INT16,
    "profile_int32": DataType.INT32,
    "surface": DataType.INT16,
    "surface_layers": DataType.FLOAT64,
    "surface_irregular": DataType.FLOAT64,
    "surface_irregular_layers": DataType.FLOAT64,
    "surface_extensions": DataType.FLOAT32,
    "surface_int32": DataType.INT32,
    "surface_placed": DataType.FLOAT64,
    "surface_absolute_placed": DataType.FLOAT64,
    "point_cloud": DataType.FLOAT32,
    "point_cloud_int16": DataType.INT16,
    "point_cloud_int32": DataType.INT32,
    "point_cloud_float64": DataType.FLOAT64,
}
#: The samples with points that were not measured.
WITH_MISSING_POINTS = {"surface", "surface_int32", "profile_int16", "profile_layers"}
STORAGES = {"binary": DataStorage.BINARY, "xml": DataStorage.XML}


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate", Path(__file__).parent / "samples" / "generate.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = _generator()
GENERATED = {revision: GENERATOR.build(revision) for revision in GENERATOR.FOLDERS}
CASES = [
    pytest.param(revision, name, suffix, id=f"{revision.name}-{name}-{suffix}")
    for revision in GENERATOR.FOLDERS
    for name in GENERATED[revision]
    for suffix in GENERATOR.storages(revision, name)
]
BINARY_CASES = [case for case in CASES if case.values[2] == "binary"]
BOTH_STORAGES = [
    (revision, name)
    for revision in GENERATOR.FOLDERS
    for name in GENERATED[revision]
    if len(GENERATOR.storages(revision, name)) == 2
]


def _path(revision: Revision, name: str, suffix: str) -> Path:
    folder: Path = GENERATOR.FOLDERS[revision]
    return folder / f"{name}_{suffix}.x3p"


def _assert_same_sample(sample: X3pFile, expected: X3pFile) -> None:
    """Compare two files, allowing the last bits of a float to differ between platforms.

    The functions that make the samples, such as ``cos``, are not exact to the last bit on every
    platform, so a sample generated on another one differs by that much.
    """
    assert type(sample) is type(expected)
    assert sample.metadata == expected.metadata
    assert sample.extensions == expected.extensions
    assert sample.revision is expected.revision
    np.testing.assert_allclose(sample.placement.rotation, expected.placement.rotation, atol=1e-12)
    np.testing.assert_allclose(sample.placement.offset, expected.placement.offset, atol=1e-12)
    for name in ("x", "y", "z"):
        axis, expected_axis = getattr(sample.header, name), getattr(expected.header, name)
        assert (axis.axis_type, axis.data_type) == (
            expected_axis.axis_type,
            expected_axis.data_type,
        )
        assert axis.increment == pytest.approx(expected_axis.increment, rel=1e-9)
        values, expected_values = stacked(sample, name), stacked(expected, name)
        if values is None or expected_values is None:
            assert values is expected_values
            continue
        # An integer value can differ by one step where a float lies on a rounding limit.
        step = axis.increment if get_data_type(axis.data_type).is_integer else 0.0
        atol = 1e-9 * float(np.nanmax(np.abs(expected_values))) + 1.001 * step
        np.testing.assert_allclose(values, expected_values, rtol=0, atol=atol, equal_nan=True)


def test_the_folders_hold_exactly_the_samples() -> None:
    for revision, folder in GENERATOR.FOLDERS.items():
        expected = {
            _path(revision, name, suffix).name
            for name in GENERATED[revision]
            for suffix in GENERATOR.storages(revision, name)
        }
        assert {path.name for path in folder.glob("*.x3p")} == expected
    assert not list(GENERATOR.FOLDER.glob("*.x3p"))


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_is_what_the_generator_makes(revision: Revision, name: str, suffix: str) -> None:
    sample = x3pio.read(_path(revision, name, suffix))
    made = GENERATED[revision][name]
    # The generated values are not rounded to the stored data type yet, a written file's are.
    _assert_same_sample(sample, type(made).loads(made.dumps(storage=STORAGES[suffix])))


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_has_its_type_layers_and_revision(
    revision: Revision, name: str, suffix: str
) -> None:
    kind, layers = KINDS[name]
    sample = x3pio.read(_path(revision, name, suffix))
    assert type(sample) is kind
    assert sample.revision is revision
    if not isinstance(sample, PointCloud):
        assert len(sample.layers) == layers


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_the_xml_of_a_sample_is_valid(revision: Revision, name: str, suffix: str) -> None:
    members = unzip(_path(revision, name, suffix).read_bytes())
    SCHEMAS[revision].validate(members["main.xml"].decode("utf-8-sig"))


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_stores_its_coordinates_as_named(
    revision: Revision, name: str, suffix: str
) -> None:
    members = unzip(_path(revision, name, suffix).read_bytes())
    assert ("bindata/data.bin" in members) == (suffix == "binary")


@pytest.mark.parametrize(("revision", "name"), BOTH_STORAGES)
def test_the_binary_and_the_xml_sample_hold_the_same_points(revision: Revision, name: str) -> None:
    binary = x3pio.read(_path(revision, name, "binary"))
    text = x3pio.read(_path(revision, name, "xml"))
    assert binary == text


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_survives_the_other_storage(revision: Revision, name: str, suffix: str) -> None:
    sample = x3pio.read(_path(revision, name, suffix))
    for storage in DataStorage:
        assert type(sample).loads(sample.dumps(storage=storage)) == sample


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_converts_to_the_other_revision(
    revision: Revision, name: str, suffix: str, tmp_path: Path
) -> None:
    other = FIRST if revision is AMENDED else AMENDED
    source = _path(revision, name, suffix)
    target = tmp_path / "converted.x3p"
    # The extensions of Amd 1:2020 cannot be converted, so they are dropped.
    arguments = ["convert", "-r", other.name, str(source), str(target)]
    assert main([*arguments, "--fix"] if revision is AMENDED else arguments) == 0
    converted = x3pio.read(target)
    assert converted.revision is other
    members = unzip(target.read_bytes())
    SCHEMAS[other].validate(members["main.xml"].decode("utf-8-sig"))
    np.testing.assert_array_equal(flat_points(converted), flat_points(x3pio.read(source)))


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_the_command_line_describes_a_sample(
    revision: Revision, name: str, suffix: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info", str(_path(revision, name, suffix))]) == 0
    kind = {Profile: "PRF", Surface: "SUR", PointCloud: "PCL"}[KINDS[name][0]]
    output = capsys.readouterr().out
    assert re.search(rf"^FeatureType\s*= {kind}$", output, re.MULTILINE)
    assert re.search(rf"^Storage\s*= {suffix}$", output, re.MULTILINE)
    assert re.search(rf"^Revision\s*= {re.escape(revision.value)}$", output, re.MULTILINE)


@pytest.mark.parametrize(("revision", "name", "suffix"), BINARY_CASES)
def test_a_sample_has_its_points(revision: Revision, name: str, suffix: str) -> None:
    sample = x3pio.read(_path(revision, name, suffix))
    points = flat_points(sample)
    assert points.shape == (stacked(sample).size, 3)
    assert np.isfinite(points[~np.isnan(points[:, 2])]).all()


@pytest.mark.parametrize("revision", GENERATOR.FOLDERS)
def test_the_extensions_of_a_sample_are_read(revision: Revision) -> None:
    sample = x3pio.read(_path(revision, "surface_extensions", "binary"))
    expected = {
        f"{vendor}/{path}": content for vendor, path, content in GENERATOR.EXTENSIONS[revision]
    }
    assert dict(sample.extensions) == expected
    vendors = {uri.split("/")[2] for uri in expected}
    assert len(vendors) == (2 if revision is AMENDED else 1)
    assert dict(type(sample).loads(sample.dumps()).extensions) == expected


def test_the_extensions_of_amd_1_cannot_be_converted() -> None:
    sample = x3pio.read(_path(AMENDED, "surface_extensions", "binary"))
    with pytest.raises(x3pio.X3pFormatError, match="cannot be converted"):
        sample.with_revision(FIRST)
    assert len(sample.with_revision(FIRST, drop_extensions=True).extensions) == 0


def test_the_extensions_of_the_first_revision_become_ids_of_amd_1() -> None:
    sample = x3pio.read(_path(FIRST, "surface_extensions", "binary"))
    converted = sample.with_revision(AMENDED)
    assert dict(converted.extensions) == dict(sample.extensions)


@pytest.mark.parametrize("revision", GENERATOR.FOLDERS)
def test_only_one_sample_has_extensions(revision: Revision) -> None:
    for name, file in GENERATED[revision].items():
        assert (len(file.extensions) > 0) == (name == "surface_extensions")


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_stores_its_data_types(revision: Revision, name: str, suffix: str) -> None:
    header = x3pio.read(_path(revision, name, suffix)).header
    expected = DATA_TYPES[name]
    assert header.z.data_type is expected
    if KINDS[name][0] is PointCloud:
        assert header.x.data_type is header.y.data_type is expected


def test_every_data_type_is_used_for_each_kind_of_data() -> None:
    for kind in (Profile, Surface, PointCloud):
        used = {DATA_TYPES[name] for name in KINDS if KINDS[name][0] is kind}
        assert used == set(DataType)


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_a_sample_has_missing_points_as_named(revision: Revision, name: str, suffix: str) -> None:
    sample = x3pio.read(_path(revision, name, suffix))
    assert bool(np.isnan(stacked(sample)).any()) == (name in WITH_MISSING_POINTS)


@pytest.mark.parametrize(("revision", "name", "suffix"), CASES)
def test_only_the_placed_samples_have_a_placement(
    revision: Revision, name: str, suffix: str
) -> None:
    sample = x3pio.read(_path(revision, name, suffix))
    assert sample.placement.is_identity == (not name.endswith("_placed"))


def test_the_placed_samples_differ_by_their_axes_only() -> None:
    revision = AMENDED
    incremental = x3pio.read(_path(revision, "surface_placed", "binary"))
    absolute = x3pio.read(_path(revision, "surface_absolute_placed", "binary"))
    assert incremental.header.x.is_incremental
    assert not absolute.header.x.is_incremental
    assert incremental.placement == absolute.placement
    np.testing.assert_allclose(flat_points(incremental), flat_points(absolute), atol=1e-12)
    view = flat_points(incremental, coordinate_system=x3pio.CoordinateSystem.VIEW)
    assert not np.allclose(view, flat_points(incremental))
