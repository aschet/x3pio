# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Files that deviate from the standard are read, as the README lists."""

from __future__ import annotations

import hashlib
import io
import re
from datetime import UTC, datetime

import numpy as np
import pytest

import x3pio
from helpers import Edit, build, drop, edit_main, flat_points, replace, rezip, stacked, sub, unzip
from x3pio import (
    DataStorage,
    DataType,
    FeatureType,
    Metadata,
    X3pFormatError,
)

#: The revision of the files of other software, with the en dash they often have.
DASHED_REVISION = "ISO5436 \N{EN DASH} 2000"


def _read(blob: bytes) -> x3pio.X3pFile:
    return x3pio.X3pFile.loads(blob)


def test_a_tolerated_file_is_written_back_as_the_standard_gives_it() -> None:
    blob = edit_main(build(), replace("<Revision>ISO25178-72:2017/DAM1", "<Revision>other"))
    x3p = _read(blob)
    assert x3p.revision is x3pio.Revision.ISO5436_2000
    assert "<Revision>ISO5436 - 2000</Revision>" in unzip(x3p.dumps())["main.xml"].decode()


def test_missing_increment_is_assumed_to_be_one() -> None:
    blob = edit_main(build(), drop(r"\s*<Increment>1\.0</Increment>"))
    x3p = _read(blob)
    assert x3p.header.z.increment == 1.0
    assert x3p.header.x.increment == 1e-6
    assert x3p.header.y.increment == 2e-6


def test_empty_offset_is_taken_as_zero() -> None:
    blob = edit_main(build(), replace("<Offset>0.0</Offset>", "<Offset />"))
    x3p = _read(blob)
    assert x3p.placement.offset[0] == 0.0


def test_missing_offset_is_not_a_deviation() -> None:
    x3p = _read(edit_main(build(), drop(r"\s*<Offset>0\.0</Offset>")))
    assert x3p.placement.offset[0] == 0.0


def test_missing_data_type_of_an_incremental_axis_is_filled_in() -> None:
    blob = edit_main(build(), drop(r"\s*<DataType>D</DataType>"))
    x3p = _read(blob)
    assert x3p.header.x.data_type is DataType.FLOAT64


def test_missing_data_type_of_an_absolute_axis_is_an_error() -> None:
    blob = edit_main(build(), drop(r"(?<=<AxisType>A</AxisType>)\s*<DataType>D</DataType>"))
    with pytest.raises(X3pFormatError, match="absolute axis needs a DataType"):
        _read(blob)


def test_other_revision_is_the_iso_25178_72_2017() -> None:
    blob = edit_main(
        build(), replace("<Revision>ISO25178-72:2017/DAM1", f"<Revision>{DASHED_REVISION}")
    )
    x3p = _read(blob)
    assert x3p.revision is x3pio.Revision.ISO5436_2000
    written = unzip(x3p.dumps())["main.xml"].decode()
    assert DASHED_REVISION not in written
    assert "<Revision>ISO5436 - 2000</Revision>" in written


@pytest.mark.parametrize("dash", ["\N{EN DASH}", "\N{EM DASH}", "\N{MINUS SIGN}"])
def test_the_revision_of_amd_1_with_another_dash_is_still_that_of_amd_1(dash: str) -> None:
    blob = edit_main(build(), replace("<Revision>ISO25178-72", f"<Revision>ISO25178{dash}72"))
    x3p = _read(blob)
    assert x3p.revision is x3pio.Revision.ISO25178_72_2017_DAM1
    assert "<Revision>ISO25178-72:2017/DAM1</Revision>" in unzip(x3p.dumps())["main.xml"].decode()


def test_missing_revision_is_the_iso_25178_72_2017() -> None:
    blob = edit_main(build(), drop(r"\s*<Revision>[^<]*</Revision>"))
    assert _read(blob).revision is x3pio.Revision.ISO5436_2000


def test_root_without_namespace() -> None:
    def edit(text: str) -> str:
        text = re.sub(r"<p:ISO5436_2[^>]*>", "<ISO5436_2>", text)
        return text.replace("</p:ISO5436_2>", "</ISO5436_2>")

    blob = edit_main(build(), edit)
    x3p = _read(blob)
    assert stacked(x3p).shape == (1, 2, 3)


def test_unknown_root_is_an_error() -> None:
    blob = edit_main(build(), replace("p:ISO5436_2", "p:Other", 2))
    with pytest.raises(X3pFormatError, match="unexpected root element"):
        _read(blob)


@pytest.mark.parametrize(
    ("closing", "extra"),
    [
        ("</Axes>", "<Origin>NW</Origin>"),
        ("</CZ>", "<Extra>1</Extra>"),
        ("</Record1>", "<Extra>1</Extra>"),
        ("</Record2>", "<Extra>1</Extra>"),
        ("</Instrument>", "<Extra>1</Extra>"),
        ("</MatrixDimension>", "<Extra>1</Extra>"),
        ("</Record3>", "<Mask><Background>#ffffff</Background></Mask>"),
        ("</DataList>", "<Other>1</Other>"),
        ("</Record4>", "<Extra>1</Extra>"),
    ],
)
def test_elements_the_schema_does_not_define_are_ignored(closing: str, extra: str) -> None:
    blob = edit_main(build(DataStorage.XML), replace(closing, extra + closing))
    assert stacked(_read(blob)).shape == (1, 2, 3)


def test_elements_after_the_last_record_are_ignored() -> None:
    blob = edit_main(build(), replace("</p:ISO5436_2>", "<Extra>1</Extra></p:ISO5436_2>"))
    assert stacked(_read(blob)).shape == (1, 2, 3)


def test_placeholder_dates_and_probing_type_are_dropped() -> None:
    date = datetime(2024, 5, 1, tzinfo=UTC)
    blob = build(metadata=x3pio.Metadata(date=date, calibration_date=date))

    def edit(text: str) -> str:
        text = sub(r"<Date>[^<]*</Date>", "<Date>N/A</Date>")(text)
        text = sub(
            r"<CalibrationDate>[^<]*</CalibrationDate>", "<CalibrationDate>x</CalibrationDate>"
        )(text)
        return text.replace("<Type>Software</Type>", "<Type>Type</Type>")

    changed = edit_main(blob, edit)
    x3p = _read(changed)
    assert x3p.metadata is not None
    assert x3p.metadata.date is None
    assert x3p.metadata.calibration_date is None
    assert x3p.metadata.probing_system.type is None


def test_placeholder_metadata_cannot_be_written_until_it_is_set() -> None:
    blob = edit_main(build(), sub(r"<Date>[^<]*</Date>", "<Date>N/A</Date>"))
    x3p = _read(blob)
    with pytest.raises(X3pFormatError, match="no date"):
        x3p.dumps()
    x3p = x3p.with_metadata(date=datetime(2024, 5, 1, tzinfo=UTC))
    assert _read(x3p.dumps()).metadata is not None


def test_probing_type_is_matched_without_regard_to_case() -> None:
    blob = edit_main(build(), replace("<Type>Software</Type>", "<Type>contacting</Type>"))
    x3p = _read(blob)
    assert x3p.metadata is not None
    assert x3p.metadata.probing_system.type is x3pio.ProbingType.CONTACTING


def test_missing_metadata_pieces_are_reported() -> None:
    blob = edit_main(build(), drop(r"\s*<Instrument>.*</Instrument>"))
    x3p = _read(blob)
    assert x3p.metadata is not None
    assert x3p.metadata.instrument == x3pio.Instrument()
    blob = edit_main(build(), drop(r"\s*<ProbingSystem>.*</ProbingSystem>"))
    assert _read(blob).metadata.probing_system.type is None  # type: ignore[union-attr]
    blob = edit_main(build(), drop(r"\s*<Serial />"))
    assert _read(blob).metadata.instrument.serial == ""  # type: ignore[union-attr]


def test_a_date_without_a_time_zone_is_read_as_local_time() -> None:
    # ISO 8601: a time without any relation to UTC is a local time.
    blob = edit_main(build(), sub(r"<Date>[^<]*</Date>", "<Date>2009-06-09T13:40:58.5</Date>"))
    x3p = _read(blob)
    assert x3p.metadata is not None
    assert x3p.metadata.date == datetime(2009, 6, 9, 13, 40, 58, 500000).astimezone()
    assert x3p.metadata.date.tzinfo is not None


def test_a_date_without_a_time_zone_can_be_written_back() -> None:
    blob = edit_main(build(), sub(r"<Date>[^<]*</Date>", "<Date>2009-06-09T13:40:58.5</Date>"))
    again = _read(_read(blob).dumps())
    assert again.metadata is not None
    assert again.metadata.date == datetime(2009, 6, 9, 13, 40, 58, 500000).astimezone()


def test_a_date_that_cannot_be_a_local_time_is_still_read() -> None:
    blob = edit_main(build(), sub(r"<Date>[^<]*</Date>", "<Date>0001-01-01T00:00:00.0</Date>"))
    x3p = _read(blob)
    assert x3p.metadata is not None
    assert x3p.metadata.date == datetime(1, 1, 1)


def test_the_calibration_date_is_read_like_the_date() -> None:
    blob = edit_main(
        build(metadata=Metadata(calibration_date=datetime(2020, 1, 1, tzinfo=UTC))),
        sub(
            r"<CalibrationDate>[^<]*</CalibrationDate>",
            "<CalibrationDate>2020-01-01T00:00:00.0</CalibrationDate>",
        ),
    )
    assert _read(blob).metadata.calibration_date == datetime(2020, 1, 1).astimezone()  # type: ignore[union-attr]


def test_checksum_file_may_have_another_name() -> None:
    members = unzip(edit_main(build(), replace("md5checksum.hex", "sum.md5")))
    members["sum.md5"] = members.pop("md5checksum.hex")
    assert stacked(_read(rezip(members))).shape == (1, 2, 3)


def test_content_nested_in_one_folder_is_found() -> None:
    members = {f"sample/{name}": content for name, content in unzip(build()).items()}
    members["__MACOSX/._sample"] = b"junk"
    x3p = _read(rezip(members))
    assert stacked(x3p).shape == (1, 2, 3)


def test_utf8_byte_order_mark_is_ignored() -> None:
    x3p = _read(edit_main(build(), lambda text: "\ufeff" + text))
    assert stacked(x3p).shape == (1, 2, 3)
    assert stacked(x3p).shape == (1, 2, 3)


def test_non_finite_floats_are_missing_points() -> None:
    blob = build(DataStorage.BINARY, z_type=DataType.FLOAT32, z_scale=1.0)
    members = unzip(blob)
    values = np.frombuffer(members["bindata/data.bin"], dtype="<f4").copy()
    values[0], values[1] = np.inf, -np.inf
    members["bindata/data.bin"] = values.tobytes()
    old = re.search(r"<MD5ChecksumPointData>([0-9a-f]+)<", members["main.xml"].decode()).group(1)  # type: ignore[union-attr]
    new = hashlib.md5(members["bindata/data.bin"], usedforsecurity=False).hexdigest()
    blob = edit_main(rezip(members), replace(old, new))
    x3p = _read(blob)
    assert np.isnan(stacked(x3p).ravel()[:3]).all()


def test_integer_data_type_of_an_incremental_axis_is_ignored() -> None:
    def edit(text: str) -> str:
        return text.replace(
            "<AxisType>I</AxisType>\n        <DataType>D</DataType>",
            "<AxisType>I</AxisType>\n        <DataType>L</DataType>",
            1,
        )

    x3p = _read(edit_main(build(), edit))
    assert x3p.header.x.data_type is DataType.INT32
    assert stacked(x3p).shape == (1, 2, 3)


def test_rotation_marginally_outside_the_range_is_clamped() -> None:
    elements = "".join(
        f"<r{row}{col}>{'1.0000001' if row == col else '0'}</r{row}{col}>"
        for row in (1, 2, 3)
        for col in (1, 2, 3)
    )
    blob = edit_main(build(), replace("</Axes>", f"<Rotation>{elements}</Rotation></Axes>"))
    x3p = _read(blob)
    assert np.array_equal(x3p.placement.rotation, np.eye(3))


def test_rotation_far_outside_the_range_is_an_error() -> None:
    elements = "".join(f"<r{row}{col}>2</r{row}{col}>" for row in (1, 2, 3) for col in (1, 2, 3))
    blob = edit_main(build(), replace("</Axes>", f"<Rotation>{elements}</Rotation></Axes>"))
    with pytest.raises(X3pFormatError, match="within"):
        _read(blob)


def test_negative_increment_is_accepted() -> None:
    blob = edit_main(
        build(), replace("<Increment>1e-06</Increment>", "<Increment>-1e-06</Increment>")
    )
    x3p = _read(blob)
    assert x3p.header.x.increment == -1e-6


def test_zero_increment_is_an_error() -> None:
    blob = edit_main(build(), replace("<Increment>1e-06</Increment>", "<Increment>0</Increment>"))
    with pytest.raises(X3pFormatError, match="must not be zero"):
        _read(blob)


def test_datum_outside_the_number_pattern_is_parsed() -> None:
    blob = edit_main(build(), replace("<Datum>2e-06</Datum>", "<Datum>2.e-06</Datum>"))
    x3p = _read(blob)
    assert stacked(x3p)[0, 0, 1] == 2e-6


def test_whitespace_around_a_datum_is_ignored() -> None:
    blob = edit_main(build(), replace("<Datum>2e-06</Datum>", "<Datum>\n 2e-06 \n</Datum>"))
    assert stacked(_read(blob))[0, 0, 1] == 2e-6


def test_feature_type_is_made_to_agree_with_the_data() -> None:
    blob = edit_main(build(), replace("<FeatureType>SUR", "<FeatureType>PRF"))
    x3p = _read(blob)
    assert x3p.feature_type is FeatureType.SURFACE

    blob = edit_main(build(), replace("<FeatureType>SUR", "<FeatureType>PCL"))
    assert _read(blob).feature_type is FeatureType.SURFACE

    pcl = build_points()
    blob = edit_main(pcl, replace("<FeatureType>PCL", "<FeatureType>SUR"))
    x3p = _read(blob)
    assert x3p.feature_type is FeatureType.POINT_CLOUD


def build_points() -> bytes:
    buffer = io.BytesIO()
    x3pio.write_points(
        buffer, np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]), storage=DataStorage.XML
    )
    return buffer.getvalue()


@pytest.mark.parametrize("storage", [DataStorage.XML, DataStorage.BINARY])
def test_a_point_cloud_with_an_incremental_axis_is_an_error(storage: DataStorage) -> None:
    buffer = io.BytesIO()
    x3pio.write_points(buffer, np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]), storage=storage)

    def edit(text: str) -> str:
        return text.replace("<AxisType>A</AxisType>", "<AxisType>I</AxisType>", 1)

    blob = edit_main(buffer.getvalue(), edit)
    with pytest.raises(X3pFormatError, match="must be absolute"):
        _read(blob)


def _with_validity(mask: int) -> bytes:
    """Build a binary point cloud file with a validity file that has the bits of ``mask``."""
    buffer = io.BytesIO()
    x3pio.write_points(buffer, np.arange(12.0).reshape(4, 3))
    members = unzip(buffer.getvalue())
    checksum = hashlib.md5(bytes([mask]), usedforsecurity=False).hexdigest()
    link = "<ValidPointsLink>bindata/valid.bin</ValidPointsLink>"
    link += f"<MD5ChecksumValidPoints>{checksum}</MD5ChecksumValidPoints>"
    members["bindata/valid.bin"] = bytes([mask])
    main = members["main.xml"].decode()
    members["main.xml"] = main.replace(
        "</MD5ChecksumPointData>", f"</MD5ChecksumPointData>{link}"
    ).encode()
    digest = hashlib.md5(members["main.xml"], usedforsecurity=False).hexdigest()
    members["md5checksum.hex"] = f"{digest} *main.xml\n".encode("ascii")
    return rezip(members)


def test_points_of_a_point_cloud_that_are_marked_invalid_are_left_out() -> None:
    x3p = _read(_with_validity(0b0101))
    assert isinstance(x3p, x3pio.PointCloud)
    assert flat_points(x3p).tolist() == [[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]
    assert _read(x3p.dumps()).layers[0] == x3p.layers[0]


def test_a_point_cloud_without_invalid_points_keeps_all_of_them() -> None:
    assert len(flat_points(_read(_with_validity(0b1111)))) == 4


def test_empty_points_of_a_point_cloud_in_text_are_left_out() -> None:
    blob = edit_main(build_points(), replace("<Datum>0.0;0.0;1.0</Datum>", "<Datum />"))
    assert flat_points(_read(blob)).tolist() == [[1.0, 2.0, 3.0]]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (replace("<AxisType>A</AxisType>", "<AxisType>I</AxisType>"), "z axis must be absolute"),
        (replace("<AxisType>A</AxisType>", "<AxisType>Q</AxisType>"), "unknown AxisType"),
        (replace("<DataType>D</DataType>", "<DataType>Q</DataType>"), "unknown DataType"),
        (replace("<FeatureType>SUR", "<FeatureType>XYZ"), "Unknown FeatureType"),
        (drop(r"\s*<Record1>.*</Record1>"), "Missing element <Record1>"),
        (drop(r"\s*<Record3>.*</Record3>"), "Missing element <Record3>"),
        (drop(r"\s*<MatrixDimension>.*</MatrixDimension>"), "exactly one of MatrixDimension"),
        (replace("<SizeX>3", "<SizeX>-3"), "must not be negative"),
        (replace("<SizeX>3", "<SizeX>three"), "not a whole number"),
        (replace("<SizeX>3", "<SizeX>4"), "holds 6 Datum elements, but 8 needed"),
        (replace("<Increment>1e-06", "<Increment>fast"), "not a number"),
        (replace("<Increment>1e-06", "<Increment>nan"), "not a finite number"),
        (replace("<Datum>2e-06", "<Datum>1;2e-06"), "holds 2 value(s), but the axes need 1"),
        (replace("<Datum>2e-06", "<Datum>two"), "not a list of numbers"),
        (
            replace(
                "<DataList>", "<DataLink><PointDataLink>x</PointDataLink></DataLink><DataList>"
            ),
            "exactly one of DataLink",
        ),
        (lambda text: text[:200], "not valid XML"),
        (replace("<Record3>", "<!DOCTYPE a [<!ENTITY b 'c'>]><Record3>"), "not valid XML"),
    ],
)
def test_uncorrectable_content_is_an_error(edit: Edit, message: str) -> None:
    with pytest.raises(X3pFormatError, match=re.escape(message)):
        _read(edit_main(build(), edit))


def test_missing_date_is_reported() -> None:
    blob = edit_main(build(), drop(r"\s*<Date>[^<]*</Date>"))
    x3p = _read(blob)
    assert x3p.metadata is not None
    assert x3p.metadata.date is None


def test_a_checksum_file_that_is_not_in_the_container_does_not_matter() -> None:
    blob = edit_main(build(), replace("md5checksum.hex", "../md5checksum.hex"))
    assert stacked(_read(blob)).shape == (1, 2, 3)
