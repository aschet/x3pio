# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

import x3pio
from helpers import array_file, build, edit_main, stacked, sub, unzip
from x3pio import (
    Instrument,
    Metadata,
    MetadataChanges,
    ProbingSystem,
    ProbingType,
    X3pFile,
    X3pFormatError,
)

DATE = datetime(2024, 5, 1, 10, 30, 15, 250000, tzinfo=timezone(timedelta(hours=2)))


def _file(metadata: Metadata | None = None) -> X3pFile:
    x3p = X3pFile.loads(build(metadata=metadata))
    return x3p


def test_default_metadata_names_x3pio_as_software() -> None:
    before = datetime.now(UTC)
    metadata = _file().metadata
    assert metadata is not None
    assert metadata.date is not None
    assert before - timedelta(seconds=5) < metadata.date < datetime.now(UTC) + timedelta(seconds=5)
    assert metadata.instrument.manufacturer == "x3pio"
    assert metadata.instrument.version == x3pio.__version__
    assert metadata.probing_system == ProbingSystem(ProbingType.SOFTWARE, "x3pio")


def test_all_metadata_roundtrips() -> None:
    metadata = Metadata(
        date=DATE,
        creator="Jane Doe, Metrology Institute",
        instrument=Instrument("Acme", "Scanner 3000", "SN-1", "fw 1.2, sw 3.4"),
        calibration_date=datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC),
        probing_system=ProbingSystem(ProbingType.NON_CONTACTING, "20x objective"),
        comment="First line\nSecond line, with <markup> & µm",
    )
    assert _file(metadata).metadata == metadata


def test_a_file_without_metadata_roundtrips() -> None:
    buffer = io.BytesIO()
    array_file(np.zeros((1, 1)), x_scale=1.0, y_scale=1.0).save(buffer)
    x3p = X3pFile.loads(buffer.getvalue())
    x3p.metadata = None
    again = X3pFile.loads(x3p.dumps())
    assert again.metadata is None
    assert "Record2" not in unzip(x3p.dumps())["main.xml"].decode()


def test_token_elements_have_their_whitespace_collapsed() -> None:
    metadata = Metadata(date=DATE, creator="  Jane \n  Doe  ")
    assert _file(metadata).metadata is not None
    assert _file(metadata).metadata.creator == "Jane Doe"  # type: ignore[union-attr]


def test_empty_creator_and_comment_are_kept() -> None:
    metadata = Metadata(date=DATE, creator="", comment="")
    stored = _file(metadata).metadata
    assert stored is not None
    assert stored.creator == ""
    assert stored.comment == ""


def test_text_that_xml_cannot_hold_is_rejected() -> None:
    with pytest.raises(X3pFormatError, match="creator"):
        build(metadata=Metadata(creator="a\x01b"))
    with pytest.raises(X3pFormatError, match="comment"):
        build(metadata=Metadata(comment="a\x00b"))


def test_metadata_without_a_required_value_cannot_be_written() -> None:
    x3p = _file()
    x3p.metadata = Metadata(date=None)
    with pytest.raises(X3pFormatError, match="no date"):
        x3p.dumps()
    x3p.metadata = Metadata(date=DATE, probing_system=ProbingSystem(type=None))
    with pytest.raises(X3pFormatError, match="no type"):
        x3p.dumps()


def test_with_metadata_changes_only_the_given_fields() -> None:
    x3p = _file(Metadata(date=DATE, creator="Jane"))
    x3p = x3p.with_metadata(creator="John", comment="Hello", serial="42")
    metadata = x3p.metadata
    assert metadata is not None
    assert (metadata.creator, metadata.comment, metadata.instrument.serial) == (
        "John",
        "Hello",
        "42",
    )
    assert metadata.date == DATE
    assert X3pFile.loads(x3p.dumps()).metadata == metadata


def test_with_metadata_covers_every_field() -> None:
    changes: MetadataChanges = {
        "date": DATE,
        "creator": "Jane",
        "calibration_date": DATE,
        "comment": "c",
        "manufacturer": "Acme",
        "model": "M",
        "serial": "S",
        "version": "V",
        "probing_type": ProbingType.CONTACTING,
        "probing_identification": "stylus",
    }
    x3p = _file()
    x3p = x3p.with_metadata(**changes)
    assert x3p.metadata == Metadata(
        DATE,
        "Jane",
        Instrument("Acme", "M", "S", "V"),
        DATE,
        ProbingSystem(ProbingType.CONTACTING, "stylus"),
        "c",
    )


def test_none_removes_optional_fields() -> None:
    x3p = _file(Metadata(date=DATE, creator="Jane", calibration_date=DATE, comment="c"))
    x3p = x3p.with_metadata(creator=None, calibration_date=None, comment=None)
    assert x3p.metadata is not None
    assert (x3p.metadata.creator, x3p.metadata.calibration_date, x3p.metadata.comment) == (
        None,
        None,
        None,
    )
    again = X3pFile.loads(x3p.dumps())
    assert again.metadata == x3p.metadata


def test_probing_type_is_set_by_its_member() -> None:
    x3p = _file()
    x3p = x3p.with_metadata(probing_type=ProbingType.CONTACTING)
    assert x3p.metadata is not None
    assert x3p.metadata.probing_system.type is ProbingType.CONTACTING


def test_the_probing_type_is_a_member_of_the_enum() -> None:
    x3p = _file()
    x3p = x3p.with_metadata(probing_type=ProbingType.NON_CONTACTING)
    assert x3p.metadata is not None
    assert x3p.metadata.probing_system.type is ProbingType.NON_CONTACTING


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(TypeError, match="Unknown metadata field"):
        _file().with_metadata(color="red")  # type: ignore[call-arg]


def test_with_metadata_on_a_file_without_metadata_creates_a_record() -> None:
    x3p = _file()
    x3p.metadata = None
    x3p = x3p.with_metadata(creator="Jane", date=DATE)
    metadata = x3p.metadata
    assert metadata is not None
    assert metadata.creator == "Jane"
    assert metadata.instrument == Instrument()
    assert metadata.probing_system.type is ProbingType.SOFTWARE
    assert X3pFile.loads(x3p.dumps()).metadata == metadata


def test_with_metadata_leaves_the_original_untouched() -> None:
    original = _file(Metadata(date=DATE, creator="Jane"))
    changed = original.with_metadata(creator="John", model="X")
    assert original.metadata is not None
    assert original.metadata.creator == "Jane"
    assert original.metadata.instrument.model == ""
    assert changed.metadata is not None
    assert (changed.metadata.creator, changed.metadata.instrument.model) == ("John", "X")
    assert np.shares_memory(stacked(changed), stacked(original))


def test_metadata_can_still_be_edited_directly() -> None:
    x3p = _file(Metadata(date=DATE))
    assert x3p.metadata is not None
    x3p.metadata.instrument.manufacturer = "Direct"
    assert X3pFile.loads(x3p.dumps()).metadata.instrument.manufacturer == "Direct"  # type: ignore[union-attr]


# --- the format of a date and time ------------------------------------------------------------

FORMAT = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+[+-]\d\d:\d\d")


def _written(**fields: object) -> dict[str, str]:
    blob = _file(Metadata(**fields)).dumps()  # type: ignore[arg-type]
    text = unzip(blob)["main.xml"].decode()
    return {
        name: match.group(1)
        for name in ("Date", "CalibrationDate")
        if (match := re.search(rf"<{name}>([^<]*)</{name}>", text))
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            datetime(2014, 7, 27, 17, 45, 9, 600000, tzinfo=timezone(timedelta(hours=2))),
            "2014-07-27T17:45:09.6+02:00",
        ),
        (
            datetime(2014, 7, 27, 17, 45, 9, tzinfo=timezone(timedelta(hours=2))),
            "2014-07-27T17:45:09.0+02:00",
        ),
        (datetime(2014, 7, 27, 17, 45, 9, 123456, tzinfo=UTC), "2014-07-27T17:45:09.123456+00:00"),
        (
            datetime(
                2014, 7, 27, 17, 45, 9, 250000, tzinfo=timezone(timedelta(hours=-5, minutes=-30))
            ),
            "2014-07-27T17:45:09.25-05:30",
        ),
        (
            datetime(2014, 7, 27, 0, 0, 0, tzinfo=timezone(timedelta(hours=14))),
            "2014-07-27T00:00:00.0+14:00",
        ),
        (datetime(1, 1, 1, 0, 0, 1, tzinfo=UTC), "0001-01-01T00:00:01.0+00:00"),
    ],
)
def test_a_date_is_written_in_the_format_of_the_standard(value: datetime, expected: str) -> None:
    written = _written(date=value, calibration_date=value)
    assert written == {"Date": expected, "CalibrationDate": expected}
    assert FORMAT.fullmatch(expected)


def test_the_example_of_the_standard_is_written_as_it_is_given() -> None:
    value = datetime.fromisoformat("2014-07-27T17:45:09.6+02:00")
    assert _written(date=value)["Date"] == "2014-07-27T17:45:09.6+02:00"


def test_the_default_date_is_in_the_format_of_the_standard() -> None:
    assert FORMAT.fullmatch(_written()["Date"])


def test_a_date_that_is_read_is_written_the_same() -> None:
    uri = "2007-04-30T13:58:02.6+02:00"
    blob = edit_main(build(), sub(r"<Date>[^<]*</Date>", f"<Date>{uri}</Date>"))
    again = X3pFile.loads(X3pFile.loads(blob).dumps())
    assert again.metadata is not None
    assert again.metadata.date == datetime.fromisoformat(uri)
    assert f"<Date>{uri}</Date>" in unzip(again.dumps())["main.xml"].decode()


def test_a_date_without_a_time_zone_is_written_as_a_local_time() -> None:
    # ISO 8601: a time without any relation to UTC is a local time.
    naive = datetime(2024, 5, 1, 10, 30, 15, 500000)
    expected = naive.astimezone().isoformat(timespec="microseconds")
    stamp, zone = expected[:-6], expected[-6:]
    expected = stamp.rstrip("0") + zone
    written = _written(date=naive, calibration_date=naive)
    assert written == {"Date": expected, "CalibrationDate": expected}
    assert FORMAT.fullmatch(written["Date"])


@pytest.mark.parametrize("field", ["date", "calibration_date"])
def test_a_date_that_cannot_be_a_local_time_cannot_be_written(field: str) -> None:
    fields = {"date": DATE, "calibration_date": DATE, field: datetime(1, 1, 1)}
    with pytest.raises(X3pFormatError, match="cannot be taken as a local time"):
        _file(Metadata(**fields)).dumps()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "offset",
    [
        timedelta(hours=14, minutes=1),
        timedelta(hours=-15),
        timedelta(seconds=30),
        timedelta(hours=1, seconds=1),
    ],
)
def test_a_time_zone_offset_must_be_whole_minutes_within_14_hours(offset: timedelta) -> None:
    with pytest.raises(X3pFormatError, match="offset"):
        _file(Metadata(date=datetime(2024, 1, 1, tzinfo=timezone(offset)))).dumps()


def test_a_bare_record_has_no_date() -> None:
    assert Metadata().date is None


def test_a_record_for_software_names_the_software() -> None:
    metadata = Metadata.for_software("analyser", "1.2")
    assert metadata.instrument == Instrument("analyser", "analyser", "", "1.2")
    assert metadata.probing_system == ProbingSystem(ProbingType.SOFTWARE, "analyser")
    assert metadata.date is not None
    assert metadata.date.tzinfo is not None
    assert Metadata.for_software("a", "1", date=DATE).date == DATE


@pytest.mark.parametrize(
    "make",
    [
        lambda metadata: array_file(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0, metadata=metadata),
        lambda metadata: array_file(np.zeros(2), x_scale=1.0, metadata=metadata),
        lambda metadata: x3pio.PointCloud.from_points(np.zeros((2, 3)), metadata=metadata),
        lambda metadata: x3pio.Surface.from_layers(
            [np.zeros((1, 2))], x_scale=1.0, y_scale=1.0, metadata=metadata
        ),
    ],
)
def test_a_new_file_dates_a_record_that_has_no_date(
    make: Callable[[Metadata], X3pFile],
) -> None:
    given = Metadata(creator="Jane")
    stored = make(given).metadata
    assert stored is not None
    assert stored.date is not None
    assert stored.creator == "Jane"
    assert given.date is None  # the record that was given is not changed
    assert make(Metadata(date=DATE)).metadata.date == DATE  # type: ignore[union-attr]
    assert X3pFile.loads(make(given).dumps()).metadata is not None
