# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""x3pio writes documents that are valid against the schema of the standard.

The schema is validated independently of x3pio's own parser, with the pure Python xmlschema
package. The deviations that x3pio tolerates while reading are checked to be deviations.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import xmlschema

from helpers import SCHEMAS, Edit, array_file, build, drop, edit_main, stacked, sub, unzip
from helpers import replace as replace_text
from x3pio import (
    Axis,
    AxisType,
    DataStorage,
    DataType,
    Header,
    Instrument,
    Metadata,
    Placement,
    PointCloud,
    ProbingSystem,
    ProbingType,
    Profile,
    Revision,
    Surface,
    X3pFile,
    X3pFormatError,
)

XSD_DIR = Path(__file__).parent / "xsd"
OFFICIAL = xmlschema.XMLSchema(XSD_DIR / "iso25178-72-fdam1.xsd")


def _main_xml(blob: bytes) -> str:
    return unzip(blob)["main.xml"].decode("utf-8-sig")


def _with_metadata(file: X3pFile, metadata: Metadata | None) -> X3pFile:
    copy = file.with_metadata()
    copy.metadata = metadata
    return copy


def _files() -> dict[str, X3pFile]:
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    date = datetime(2024, 5, 1, 10, 30, 15, tzinfo=UTC)
    full = Metadata(
        date=date,
        creator="Jane Doe",
        instrument=Instrument("Acme", "Scanner", "", "1.0"),
        calibration_date=date,
        probing_system=ProbingSystem(ProbingType.CONTACTING, "stylus"),
        comment="A <comment> & more",
    )
    surface = array_file(np.array([[1.0, np.nan], [3.0, 4.0]]) * 1e-6, x_scale=1e-6, y_scale=2e-6)
    absolute = Surface.from_stacked(
        Header(
            x=Axis(AxisType.ABSOLUTE, DataType.INT16, 1e-6),
            y=Axis(AxisType.ABSOLUTE, DataType.FLOAT32, 1.0),
            z=Axis(AxisType.ABSOLUTE, DataType.INT32, 1e-9),
        ),
        np.arange(4.0).reshape(1, 2, 2) * 1e-6,
        x=np.arange(4.0).reshape(1, 2, 2) * 1e-6,
        y=np.ones((1, 2, 2)),
        placement=Placement(rotation, offset=[0.0, 0.0, 1.0]),
    )
    extended = array_file(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0)
    extended.extensions["http://www.vendor.com/mypath/ext.xml"] = b"<x/>"
    extended.extensions["https://example.org/other.bin"] = b"1"
    return {
        "surface": surface,
        "layers": array_file(np.zeros((2, 2, 3)), x_scale=1.0, y_scale=1.0),
        "profile": array_file(np.array([1.0, 2.0, np.nan]), x_scale=1.0),
        "cloud": PointCloud.from_points(np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
        "absolute_axes_with_rotation": absolute,
        "full_metadata": _with_metadata(surface, full),
        "no_metadata": _with_metadata(surface, None),
        "irregular_surface": Surface.from_points(
            np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]], [[0.0, 1.5, 3.0], [1.2, 1.5, np.nan]]])
            * 1e-3,
            data_type=DataType.INT16,
        ),
        "profile_along_a_path": Profile.from_points(
            np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 1.0, 3.0]]) * 1e-3
        ),
        "profile_of_layers": Profile.from_array(np.zeros((2, 4)), x_scale=1e-6),
        "vendor_extensions": extended,
    }


FILES = _files()


def _in(name: str, revision: Revision) -> X3pFile:
    """Return a file in a revision, without extensions, which cannot be mapped to ISO5436_2000."""
    return FILES[name].with_revision(revision, drop_extensions=True)


@pytest.mark.parametrize("revision", list(Revision))
@pytest.mark.parametrize("storage", list(DataStorage))
@pytest.mark.parametrize("name", list(FILES))
def test_written_main_xml_is_valid(name: str, storage: DataStorage, revision: Revision) -> None:
    SCHEMAS[revision].validate(_main_xml(_in(name, revision).dumps(storage=storage)))


@pytest.mark.parametrize("revision", list(Revision))
@pytest.mark.parametrize("z_type", list(DataType))
def test_written_main_xml_is_valid_for_every_z_type(z_type: DataType, revision: Revision) -> None:
    file = _in("surface", revision).with_z_type(z_type)
    SCHEMAS[revision].validate(_main_xml(file.dumps()))
    SCHEMAS[revision].validate(_main_xml(file.dumps(storage=DataStorage.XML)))


def test_the_vendor_files_of_the_iso_25178_72_2017_are_valid() -> None:
    file = _in("surface", Revision.ISO5436_2000)
    file.extensions.add("http://www.example.org/format", "intensity.dat", b"1")
    file.extensions.add("http://www.example.org/format", "sub/color.dat", b"2")
    SCHEMAS[Revision.ISO5436_2000].validate(_main_xml(file.dumps()))


def test_several_ids_are_valid_only_in_amd_1() -> None:
    xml = _main_xml(FILES["vendor_extensions"].dumps())
    assert xml.count("<VendorSpecificID>") == 2
    assert SCHEMAS[Revision.ISO25178_72_2017_DAM1].is_valid(xml)
    assert not SCHEMAS[Revision.ISO5436_2000].is_valid(xml)


def test_the_calibration_date_comes_from_the_creation_date_where_the_schema_requires_it() -> None:
    # The schema of the standard requires the element, which Amd 1:2020 makes optional.
    surface = FILES["surface"]
    assert surface.metadata is not None
    assert surface.metadata.calibration_date is None

    converted = surface.with_revision(Revision.ISO5436_2000)
    assert converted.metadata is not None
    assert converted.metadata.calibration_date == surface.metadata.date
    assert surface.metadata.calibration_date is None  # the file that was converted is as it was
    SCHEMAS[Revision.ISO5436_2000].validate(_main_xml(converted.dumps()))

    # A file that is made in the revision gets it when it is written, and stays as it is.
    date = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    built = array_file(
        np.zeros((1, 2)),
        x_scale=1.0,
        y_scale=1.0,
        metadata=Metadata(date=date),
        revision=Revision.ISO5436_2000,
    )
    SCHEMAS[Revision.ISO5436_2000].validate(_main_xml(built.dumps()))
    written = X3pFile.loads(built.dumps()).metadata
    assert written is not None
    assert written.calibration_date == date
    assert built.metadata is not None
    assert built.metadata.calibration_date is None

    amended = _main_xml(surface.dumps())
    assert "<CalibrationDate>" not in amended
    SCHEMAS[Revision.ISO25178_72_2017_DAM1].validate(amended)


def test_a_calibration_date_that_is_set_is_written_as_it_is() -> None:
    date = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    for revision in Revision:
        file = FILES["surface"].with_revision(revision).with_metadata(calibration_date=date)
        written = X3pFile.loads(file.dumps()).metadata
        assert written is not None
        assert written.calibration_date == date


def test_a_file_without_metadata_needs_no_calibration_date() -> None:
    file = FILES["no_metadata"].with_revision(Revision.ISO5436_2000)
    xml = _main_xml(file.dumps())
    assert "<Record2>" not in xml
    SCHEMAS[Revision.ISO5436_2000].validate(xml)


def test_the_schema_check_is_not_vacuous() -> None:
    xml = _main_xml(build())
    assert not OFFICIAL.is_valid(xml.replace("<SizeX>3</SizeX>", ""))


def _rotation(value: str) -> Edit:
    elements = "".join(
        f"<r{row}{col}>{value if row == col else '0'}</r{row}{col}>"
        for row in (1, 2, 3)
        for col in (1, 2, 3)
    )
    return replace_text("</Axes>", f"<Rotation>{elements}</Rotation></Axes>")


#: Deviations that a schema can express, which the schema rejects and x3pio corrects.
TOLERATED = {
    "missing z increment": drop(r"\s*<Increment>1\.0</Increment>"),
    "empty offset": replace_text("<Offset>0.0</Offset>", "<Offset />"),
    "missing data type": drop(r"\s*<DataType>D</DataType>"),
    "placeholder date": sub(r"<Date>[^<]*</Date>", "<Date>N/A</Date>"),
    "placeholder probing type": replace_text("<Type>Software</Type>", "<Type>Type</Type>"),
    "other case of probing type": replace_text("<Type>Software</Type>", "<Type>software</Type>"),
    "datum outside the pattern": replace_text("<Datum>2e-06</Datum>", "<Datum>2.e-06</Datum>"),
    "missing record 4": drop(r"\s*<Record4>.*</Record4>"),
    "missing instrument": drop(r"\s*<Instrument>.*</Instrument>"),
    "rotation slightly outside": _rotation("1.0000005"),
}


@pytest.mark.parametrize("name", list(TOLERATED))
def test_tolerated_deviation(name: str) -> None:
    blob = edit_main(build(metadata=Metadata(comment="Hello")), TOLERATED[name])
    xml = _main_xml(blob)
    assert not OFFICIAL.is_valid(xml)
    assert stacked(X3pFile.loads(blob)).shape == (1, 2, 3)


def test_missing_data_checksum_is_tolerated() -> None:
    blob = edit_main(
        build(DataStorage.BINARY),
        sub(r"\s*<MD5ChecksumPointData>[^<]*</MD5ChecksumPointData>", ""),
    )
    xml = _main_xml(blob)
    assert not OFFICIAL.is_valid(xml)


#: Content that the schema rejects and x3pio refuses to read.
REJECTED = {
    "unknown axis type": replace_text("<AxisType>A</AxisType>", "<AxisType>Q</AxisType>"),
    "unknown data type": replace_text("<DataType>D</DataType>", "<DataType>Q</DataType>"),
    "unknown feature type": replace_text("<FeatureType>SUR", "<FeatureType>XYZ"),
    "missing record 1": drop(r"\s*<Record1>.*</Record1>"),
    "missing record 3": drop(r"\s*<Record3>.*</Record3>"),
    "negative size": replace_text("<SizeX>3", "<SizeX>-3"),
    "size that is not a number": replace_text("<SizeX>3", "<SizeX>three"),
    "increment that is not a number": replace_text("<Increment>1e-06", "<Increment>fast"),
    "rotation far outside": _rotation("2"),
    "missing axis": drop(r"\s*<CY>.*</CY>"),
}


@pytest.mark.parametrize("name", list(REJECTED))
def test_rejected_content(name: str) -> None:
    blob = edit_main(build(), REJECTED[name])
    xml = _main_xml(blob)
    assert not OFFICIAL.is_valid(xml)
    with pytest.raises(X3pFormatError):
        X3pFile.loads(blob)
