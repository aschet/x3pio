# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Regression tests against the sample files of the openGPS x3p library.

See tests/interop_samples/README.md for provenance and how the files differ from Amd 1:2020.
Unlike nearly every other test in this suite, which exercises x3pio's reader against x3pio's own
writer, these catch a case where x3pio's interpretation of the standard diverges from how the
openGPS reference implementation produces files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xmlschema

import x3pio
from helpers import SCHEMAS, absolute, container_from_main, flat_points, stacked, unzip
from x3pio import AxisType, DataStorage, DataType, FeatureType, Revision, Surface, X3pFile

SAMPLES_DIR = Path(__file__).parent / "interop_samples"
OPENGPS_DIR = SAMPLES_DIR / "opengps"
XSD_DIR = Path(__file__).parent / "xsd"
OFFICIAL = xmlschema.XMLSchema(XSD_DIR / "iso25178-72-fdam1.xsd")

NAN = float("nan")
#: The heights of samples 1 to 3, in metres, and the missing point of the 8th position.
HEIGHTS = np.array(
    [
        0.486219120804151,
        0.00346341436648013,
        -0.80836857168283,
        -0.579793099037002,
        0.85762202739331,
        1.04759602566142,
        1.01879225277798,
        NAN,
        0.823683772970184,
        0.797872489327661,
        -0.557459388341694,
        -0.23324785884922,
        0.675397146760858,
        0.420737549074718,
        0.64206924811095,
        -0.215696638464903,
    ]
)
#: The stored integers of sample 4; its z axis has the increment 0.001 and the offset -0.5.
SAMPLE4_STORED = [
    986,
    503,
    -308,
    -80,
    1358,
    1548,
    1519,
    0,
    1324,
    1298,
    -57,
    267,
    1175,
    921,
    1142,
    284,
]
SAMPLE4_HEIGHTS = np.array(SAMPLE4_STORED, dtype=float) * 0.001
SAMPLE4_HEIGHTS[7] = NAN

ALL = sorted(path.name for path in OPENGPS_DIR.glob("*.x3p"))
SAMPLE4 = ["ISO5436-sample4.x3p", "ISO5436-sample4_bin.x3p"]
SAMPLE1 = ["ISO5436-sample1.x3p", "ISO5436-sample1_bin.x3p"]


def _main_xml(name: str) -> str:
    return unzip((OPENGPS_DIR / name).read_bytes())["main.xml"].decode("utf-8-sig")


def _read(name: str) -> X3pFile:
    return x3pio.read(OPENGPS_DIR / name)


def test_all_samples_are_present() -> None:
    assert len(ALL) == 6


@pytest.mark.parametrize("name", SAMPLE4)
def test_sample4_is_valid_against_the_schema_of_the_standard(name: str) -> None:
    assert OFFICIAL.is_valid(_main_xml(name))


@pytest.mark.parametrize("name", ALL)
def test_every_sample_is_valid_against_the_schema_of_the_iso_25178_72_2017(
    name: str,
) -> None:
    SCHEMAS[Revision.ISO5436_2000].validate(_main_xml(name))


@pytest.mark.parametrize("name", ALL)
def test_every_sample_is_a_file_of_the_iso_25178_72_2017(name: str) -> None:
    assert _read(name).revision is Revision.ISO5436_2000


@pytest.mark.parametrize("name", SAMPLE4)
def test_sample4_is_read_with_the_scale_and_offset_of_the_z_axis(name: str) -> None:
    x3p = _read(name)

    assert x3p.feature_type is FeatureType.SURFACE
    assert stacked(x3p).shape == (1, 4, 4)
    assert x3p.header.z.data_type is DataType.INT16
    assert x3p.header.z.increment == 0.001
    assert x3p.placement.offset[2] == -0.5
    np.testing.assert_allclose(stacked(x3p).ravel(), SAMPLE4_HEIGHTS, atol=1e-15, equal_nan=True)
    assert x3p.raw_data()["z"].tolist() == SAMPLE4_STORED
    # The offset is part of the global coordinates only.
    np.testing.assert_allclose(
        flat_points(x3p)[:, 2], SAMPLE4_HEIGHTS - 0.5, atol=1e-15, equal_nan=True
    )
    # Its revision is the one before Amd 1:2020, which is not a deviation.


def test_sample4_xml_and_binary_hold_the_same_data() -> None:
    xml, binary = _read(SAMPLE4[0]), _read(SAMPLE4[1])
    assert xml.storage is DataStorage.XML
    assert binary.storage is DataStorage.BINARY
    assert xml == binary


def test_sample4_has_the_axes_of_a_regular_grid() -> None:
    x3p = Surface.open(OPENGPS_DIR / SAMPLE4[0])
    assert x3p.is_regular_grid
    assert x3p.layers[0].x_values[:3].tolist() == pytest.approx([0.0, 0.016016, 0.032032])
    assert x3p.metadata is not None
    assert x3p.metadata.instrument.manufacturer == "NanoFocus AG"
    assert x3p.metadata.instrument.model == "µSurf X"
    assert x3p.metadata.date is not None
    assert x3p.metadata.date.utcoffset() is not None


@pytest.mark.parametrize("name", SAMPLE1)
def test_sample1_lacks_the_z_increment_which_is_assumed(name: str) -> None:
    x3p = _read(name)

    np.testing.assert_allclose(stacked(x3p).ravel(), HEIGHTS, atol=1e-15, equal_nan=True)
    assert x3p.header.z.increment == 1.0
    assert not OFFICIAL.is_valid(_main_xml(name))


def test_sample1_xml_and_binary_hold_the_same_data() -> None:
    xml, binary = _read(SAMPLE1[0]), _read(SAMPLE1[1])
    assert np.array_equal(stacked(xml), stacked(binary), equal_nan=True)
    assert xml.metadata == binary.metadata
    # Only the XML file spells out the unit rotation, which is the same as none.
    assert xml.placement.is_identity
    assert binary.placement.is_identity


def test_sample2_is_a_point_cloud_with_absolute_axes() -> None:
    x3p = _read("ISO5436-sample2.x3p")

    assert x3p.feature_type is FeatureType.POINT_CLOUD
    assert stacked(x3p).shape == (16,)
    assert stacked(x3p, "x") is not None
    assert stacked(x3p, "y") is not None
    points = flat_points(x3p)
    # The offsets are 0.1, 2.0 and 2.0; the coordinates are 0e-6 ... 3e-6 in x.
    np.testing.assert_allclose(points[:2, 0], [0.1, 0.100001])
    np.testing.assert_allclose(points[:5, 1], [2.0, 2.0, 2.0, 2.0, 2.000001])
    # A list cannot hold a missing point, so the 8th point has a height here.
    assert np.isfinite(points).all()
    known = np.isfinite(HEIGHTS)
    np.testing.assert_allclose(points[known, 2], HEIGHTS[known] + 2.0, atol=1e-15)


def test_sample3_has_an_absolute_integer_y_axis() -> None:
    x3p = _read("ISO5436-sample3.x3p")

    assert x3p.header.x.axis_type is AxisType.INCREMENTAL
    assert x3p.header.y.axis_type is AxisType.ABSOLUTE
    assert x3p.header.y.data_type is DataType.INT16
    assert stacked(x3p, "y") is not None
    # Stored are the row numbers 1 to 4; the increment 2 and the offset 1 give 3, 5, 7, 9.
    assert absolute(x3p, "y")[0, :, 0].tolist() == [2.0, 4.0, 6.0, 8.0]
    points = flat_points(x3p)
    assert sorted({round(value, 6) for value in points[:, 1][np.isfinite(points[:, 1])]}) == [
        3.0,
        5.0,
        7.0,
        9.0,
    ]
    np.testing.assert_allclose(points[:3, 0], [0.1, 0.116016, 0.132032])


@pytest.mark.parametrize("name", ALL)
def test_converting_a_sample_gives_a_valid_file_with_the_same_data(name: str) -> None:
    read = _read(name)
    for revision in Revision:
        original = read.with_revision(revision)
        for storage in DataStorage:
            converted = original.dumps(storage=storage)
            SCHEMAS[revision].validate(unzip(converted)["main.xml"].decode())
            again = X3pFile.loads(converted)
            assert again.revision is revision
            np.testing.assert_allclose(
                stacked(again), stacked(original), atol=1e-15, equal_nan=True
            )
            np.testing.assert_allclose(
                flat_points(again), flat_points(original), atol=1e-15, equal_nan=True
            )


def _example_of_the_standard() -> bytes:
    return (SAMPLES_DIR / "iso25178-72-fdam1.xml").read_bytes()


def test_example_of_the_standard_is_read_without_any_correction() -> None:
    """The sample main.xml of Amd 1:2020 is a conforming document."""
    x3p = X3pFile.loads(container_from_main(_example_of_the_standard()))

    assert x3p.feature_type is FeatureType.SURFACE
    assert stacked(x3p).shape == (1, 4, 4)
    assert x3p.header.x.increment == 1.6016e-6
    assert x3p.header.y.increment == 1.6016e-6
    assert x3p.header.z.increment == 1.0
    assert x3p.placement.is_identity  # the unit rotation and the zero offsets of the example
    expected = np.array(
        [4.86219120804151, 3.46341436648013, -8.08368571682830, -5.79793099037002,
         8.57622027393310, 1.04759602566142, 1.01879225277798, NAN,
         8.23683772970184, 7.97872489327661, -5.57459388341694, -2.33247858849220,
         6.75397146760858, 4.20737549074718, 6.42069248110950, -2.15696638464903]
    ) * 1e-6  # fmt: skip
    np.testing.assert_allclose(stacked(x3p).ravel(), expected, rtol=1e-14, equal_nan=True)
    metadata = x3p.metadata
    assert metadata is not None
    assert metadata.creator == "Name of measuring person"
    assert metadata.instrument.manufacturer == "Sample Metrology Inc"
    assert metadata.instrument.serial == "12345abc"
    assert metadata.date is not None
    assert metadata.date.isoformat() == "2007-04-30T13:58:02.600000+02:00"
    assert metadata.probing_system.type is x3pio.ProbingType.NON_CONTACTING


def test_example_of_the_standard_is_written_back_unchanged_in_content() -> None:
    x3p = X3pFile.loads(container_from_main(_example_of_the_standard()))
    again = X3pFile.loads(x3p.dumps())
    assert again == x3p


def test_example_of_the_standard_is_valid_against_the_schema() -> None:
    OFFICIAL.validate(_example_of_the_standard().decode("utf-8"))
