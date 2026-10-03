# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The two revisions of the standard, ISO 25178-72:2017 and Amd 1:2020.

They differ in the ``Revision`` text and in how vendor specific extensions are held: a single
``VendorSpecificID`` and files of any name in the standard, an ID for each file in Amd 1:2020.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import x3pio
from helpers import SCHEMAS, array_file, build, edit_main, rezip, stacked, unzip
from x3pio import (
    DataStorage,
    Header,
    Metadata,
    PointCloud,
    Revision,
    Surface,
    VendorExtensions,
    X3pFile,
    X3pFormatError,
)
from x3pio.cli import main
from x3pio.codec.archive import PackedMember

BASE = Revision.ISO5436_2000
AMENDED = Revision.ISO25178_72_2017_DAM1

VENDOR = "nanofocus.de"
FILES = {"intensity.dat": b"\x01\x02", "color.dat": b"\x03", "metadata.npsx": b"<x/>"}


DATE = datetime(2024, 5, 1, 10, 30, 15, tzinfo=UTC)


def _file(revision: Revision = AMENDED) -> X3pFile:
    return array_file(
        np.zeros((1, 2)),
        x_scale=1.0,
        y_scale=1.0,
        metadata=Metadata(date=DATE, calibration_date=DATE),
        revision=revision,
    )


def _with_id(blob: bytes, *uris: str) -> bytes:
    ids = "".join(f"<VendorSpecificID>{uri}</VendorSpecificID>" for uri in uris)
    return edit_main(blob, lambda text: text.replace("</p:ISO5436_2>", ids + "</p:ISO5436_2>"))


def _read_base(uris: list[str], members: dict[str, bytes]) -> X3pFile:
    """Read a file of ISO 25178-72:2017 with the given IDs and extra files."""
    blob = build(revision=BASE)
    files = unzip(_with_id(blob, *uris) if uris else blob)
    files.update(members)
    return X3pFile.loads(rezip(files))


def _keys(files: dict[str, bytes], vendor: str = VENDOR) -> dict[str, bytes]:
    """Return the files as the extensions hold them: by the vendor, a "/" and the path."""
    return {f"{vendor}/{name}": data for name, data in files.items()}


def _ids(x3p: X3pFile) -> list[str]:
    """Return the VendorSpecificID that is written for the file."""
    return x3p.extensions._parts()[0]


# --- the revision --------------------------------------------------------------------------------


def test_the_revisions_are_named_by_their_revision_text() -> None:
    assert Revision.ISO5436_2000.value == "ISO5436 - 2000"
    assert Revision.ISO25178_72_2017_DAM1.value == "ISO25178-72:2017/DAM1"


def test_only_amd_1_gives_every_file_an_id() -> None:
    assert AMENDED.has_vendor_ids
    assert not BASE.has_vendor_ids


@pytest.mark.parametrize(
    ("text", "revision"),
    [
        ("ISO25178-72:2017/DAM1", AMENDED),
        ("  ISO25178-72:2017/DAM1\n", AMENDED),
        ("ISO5436 - 2000", BASE),
        ("ISO5436 \N{EN DASH} 2000", BASE),
        ("something else", BASE),
        ("", BASE),
    ],
)
def test_the_text_of_the_revision_gives_the_revision(text: str, revision: Revision) -> None:
    assert Revision.resolve(text) is revision


def test_new_files_follow_amd_1() -> None:
    assert Surface.from_stacked(Header(), np.zeros((1, 1, 2))).revision is AMENDED
    assert array_file(np.zeros(2), x_scale=1.0).revision is AMENDED
    assert PointCloud.from_points(np.zeros((2, 3))).revision is AMENDED
    assert x3pio.read(io.BytesIO(build())).revision is AMENDED


def test_every_way_to_make_a_file_takes_the_revision(tmp_path: Path) -> None:
    surface = array_file(np.zeros(2), x_scale=1.0, revision=BASE)
    cloud = PointCloud.from_points(np.zeros((2, 3)), revision=BASE)
    assert surface.revision is cloud.revision is BASE
    path = tmp_path / "a.x3p"
    x3pio.write(path, np.zeros(2), x_scale=1.0, revision=BASE)
    assert x3pio.read(path).revision is BASE
    x3pio.write_points(path, np.zeros((2, 3)), revision=BASE)
    assert x3pio.read(path).revision is BASE


def test_the_revision_is_written_as_the_revision() -> None:
    for revision in Revision:
        text = unzip(_file(revision).dumps())["main.xml"].decode()
        assert f"<Revision>{revision.value}</Revision>" in text


def test_the_revision_is_part_of_the_comparison() -> None:
    assert _file(AMENDED) != _file(BASE)
    assert _file(BASE) == _file(BASE)


def test_a_file_has_the_extensions_of_its_revision() -> None:
    assert _file(AMENDED).extensions._revision is AMENDED
    assert _file(BASE).extensions._revision is BASE


def test_empty_extensions_of_another_revision_take_the_revision_of_the_file() -> None:
    base = Surface.from_stacked(Header(), np.zeros((1, 1, 2)), revision=BASE)
    assert base.extensions._revision is BASE
    amended = Surface.from_stacked(Header(), np.zeros((1, 1, 2)), extensions=VendorExtensions(BASE))
    assert amended.extensions._revision is AMENDED


def test_the_revision_belongs_to_the_file_and_not_to_its_layers() -> None:
    first = Surface.from_array(np.zeros((2, 2)), x_scale=1.0, y_scale=1.0, revision=BASE)
    second = Surface.from_array(np.ones((2, 2)), x_scale=1.0, y_scale=1.0)
    assert first.layers[0].header == second.layers[0].header
    mixed = Surface.from_layers(
        [first.layers[0], second.layers[0]]
    )  # axes agree, revisions do not matter
    assert mixed.revision is AMENDED
    assert Surface.from_layers([first.layers[0], second.layers[0]], revision=BASE).revision is BASE
    assert first.with_layers([second.layers[0]]).revision is BASE  # the file keeps its own


def test_extensions_of_another_revision_are_not_written() -> None:
    x3p = _file(AMENDED)
    x3p.extensions["http://www.vendor.com/a.xml"] = b"<a/>"
    x3p = Surface.from_stacked(x3p.header, stacked(x3p), revision=BASE, extensions=x3p.extensions)
    with pytest.raises(X3pFormatError, match="with_revision"):
        x3p.dumps()


# --- reading a file of ISO 25178-72:2017 ----------------------------------------


def test_the_files_of_the_single_id_keep_their_names() -> None:
    x3p = _read_base([VENDOR], FILES)
    assert x3p.revision is BASE
    assert _ids(x3p) == [VENDOR]
    assert dict(x3p.extensions) == _keys(FILES)


def test_the_id_may_be_a_full_uri() -> None:
    vendor = "http://www.example.org/examplefileformat"
    x3p = _read_base([vendor], {"notes_of_sample.txt": b"text"})
    assert _ids(x3p) == [vendor]
    assert dict(x3p.extensions) == _keys({"notes_of_sample.txt": b"text"}, vendor)


def test_files_in_folders_keep_their_path() -> None:
    x3p = _read_base([VENDOR], {"sub/dir/a.dat": b"1", "b\\c.dat": b"2"})
    assert dict(x3p.extensions) == _keys({"sub/dir/a.dat": b"1", "b/c.dat": b"2"})


def test_the_files_of_the_format_are_not_extension_files() -> None:
    def edit(text: str) -> str:
        text = text.replace("bindata/data.bin", "points.bin").replace(
            "bindata/valid.bin", "mask.bin"
        )
        text = text.replace("md5checksum.hex", "sum.md5")
        return text.replace(
            "</p:ISO5436_2>", f"<VendorSpecificID>{VENDOR}</VendorSpecificID></p:ISO5436_2>"
        )

    blob = build(DataStorage.BINARY, revision=BASE, z_type=x3pio.DataType.INT16)
    members = unzip(edit_main(blob, edit))
    members["points.bin"] = members.pop("bindata/data.bin")
    members["mask.bin"] = members.pop("bindata/valid.bin")
    members["sum.md5"] = members.pop("md5checksum.hex")
    members["extra.dat"] = b"1"
    x3p = X3pFile.loads(rezip(members))
    assert dict(x3p.extensions) == _keys({"extra.dat": b"1"})
    assert np.isnan(stacked(x3p)[0, 0, 2])  # the data and validity files were used as they are


def test_the_leftovers_of_other_systems_are_not_extension_files() -> None:
    files = {"a.dat": b"1", "__MACOSX/._a.dat": b"x", ".DS_Store": b"x", "Thumbs.db": b"x"}
    assert list(_read_base([VENDOR], files).extensions) == [f"{VENDOR}/a.dat"]


def test_a_file_nested_in_a_folder_is_read_too() -> None:
    members = {
        f"sample/{name}": data
        for name, data in unzip(_with_id(build(revision=BASE), VENDOR)).items()
    }
    members["sample/a.dat"] = b"1"
    members["readme.txt"] = b"beside the folder"
    assert dict(X3pFile.loads(rezip(members)).extensions) == _keys({"a.dat": b"1"})


def test_the_files_stay_packed_until_they_are_asked_for() -> None:
    payload = bytes(range(256)) * 64
    x3p = _read_base([VENDOR], {"big.dat": payload, "other.dat": b"ok"})
    assert all(isinstance(stored, PackedMember) for stored in x3p.extensions._store.files.values())
    assert x3p.extensions[f"{VENDOR}/big.dat"] == payload


def test_files_without_an_id_are_ignored() -> None:
    x3p = _read_base([], {"a.dat": b"1"})
    assert len(x3p.extensions) == 0
    assert _ids(x3p) == []


def test_an_id_without_files_is_kept() -> None:
    x3p = _read_base([VENDOR], {})
    assert len(x3p.extensions) == 0
    assert _ids(x3p) == [VENDOR]
    with pytest.raises(X3pFormatError, match="single vendor"):
        x3p.extensions.add("other.example", "a.dat", b"1")


def test_a_file_with_several_ids_is_a_file_of_amd_1() -> None:
    x3p = _read_base([VENDOR, "other.example"], {"a.dat": b"1"})
    assert x3p.revision is AMENDED
    assert dict(x3p.extensions) == {VENDOR: None, "other.example": None}


def test_a_file_with_one_id_that_names_a_file_of_amd_1_is_read_by_its_revision() -> None:
    uri = "http://www.vendor.com/mypath/a.xml"
    amended = unzip(_with_id(build(), uri))
    amended["www\\vendor\\com\\mypath\\a.xml"] = b"<a/>"
    x3p = X3pFile.loads(rezip(amended))
    assert x3p.revision is AMENDED
    assert dict(x3p.extensions) == {uri: b"<a/>"}


def test_a_revision_that_is_not_known_is_the_iso_25178_72_2017() -> None:
    blob = edit_main(
        _with_id(build(), VENDOR), lambda text: text.replace("ISO25178-72:2017/DAM1", "future")
    )
    members = unzip(blob)
    members["a.dat"] = b"1"
    x3p = X3pFile.loads(rezip(members))
    assert x3p.revision is BASE
    assert dict(x3p.extensions) == _keys({"a.dat": b"1"})


def test_a_damaged_file_is_reported_when_it_is_asked_for() -> None:
    members = unzip(_with_id(build(revision=BASE), VENDOR))
    members["a.dat"] = b"some content that is stored deflated" * 20
    blob = bytearray(rezip(members))
    with zipfile.ZipFile(io.BytesIO(bytes(blob))) as archive:
        info = archive.getinfo("a.dat")
    start = info.header_offset + 30 + len(info.filename) + len(info.extra) + info.compress_size // 2
    blob[start] ^= 0xFF
    x3p = X3pFile.loads(bytes(blob))
    assert list(x3p.extensions) == [f"{VENDOR}/a.dat"]
    with pytest.raises(X3pFormatError, match=r"a\.dat"):
        x3p.extensions[f"{VENDOR}/a.dat"]


# --- writing a file of ISO 25178-72:2017 ----------------------------------------


def test_the_single_id_and_the_files_are_written_as_they_are() -> None:
    x3p = _read_base([VENDOR], {**FILES, "sub/dir/a.dat": b"1"})
    blob = x3p.dumps()
    members = unzip(blob)
    assert {name: members[name] for name in (*FILES, "sub/dir/a.dat")} == {
        **FILES,
        "sub/dir/a.dat": b"1",
    }
    text = members["main.xml"].decode()
    assert text.count("<VendorSpecificID>") == 1
    assert f"<VendorSpecificID>{VENDOR}</VendorSpecificID>" in text
    assert "<Revision>ISO5436 - 2000</Revision>" in text
    SCHEMAS[BASE].validate(text)
    assert X3pFile.loads(blob) == x3p


@pytest.mark.parametrize("storage", list(DataStorage))
def test_a_file_of_the_standard_is_the_same_when_it_is_read_again(storage: DataStorage) -> None:
    x3p = _read_base([VENDOR], FILES)
    assert X3pFile.loads(x3p.dumps(storage=storage)) == x3p


def test_an_id_without_files_is_written() -> None:
    x3p = _read_base([VENDOR], {})
    assert (
        f"<VendorSpecificID>{VENDOR}</VendorSpecificID>" in unzip(x3p.dumps())["main.xml"].decode()
    )


def test_the_vendor_of_the_first_file_is_the_id_that_is_written() -> None:
    x3p = _file(BASE)
    assert _ids(x3p) == []
    x3p.extensions.add("http://www.example.org/format", "intensity.dat", b"1")
    assert _ids(x3p) == ["http://www.example.org/format"]
    assert list(x3p.extensions) == ["http://www.example.org/format/intensity.dat"]
    members = unzip(x3p.dumps())
    assert members["intensity.dat"] == b"1"
    assert members["main.xml"].decode().count("<VendorSpecificID>") == 1
    again = X3pFile.loads(x3p.dumps())
    assert again == x3p


def test_a_file_of_another_vendor_is_refused() -> None:
    x3p = _file(BASE)
    x3p.extensions.add(VENDOR, "a.dat", b"1")
    with pytest.raises(X3pFormatError, match="single vendor"):
        x3p.extensions.add("other.example", "b.dat", b"2")
    assert dict(x3p.extensions) == _keys({"a.dat": b"1"})
    assert _ids(x3p) == [VENDOR]


def test_the_vendor_is_compared_without_a_trailing_slash_and_whitespace() -> None:
    x3p = _file(BASE)
    x3p.extensions.add(f" {VENDOR}/ ", "a.dat", b"1")
    x3p.extensions.add(VENDOR, "b.dat", b"2")
    assert list(x3p.extensions) == [f"{VENDOR}/a.dat", f"{VENDOR}/b.dat"]
    assert _ids(x3p) == [f"{VENDOR}/"]  # as given, without the whitespace


def test_the_vendor_stays_when_the_last_file_is_removed_and_clear_forgets_it() -> None:
    x3p = _file(BASE)
    x3p.extensions.add(VENDOR, "a.dat", b"1")
    del x3p.extensions[f"{VENDOR}/a.dat"]
    assert len(x3p.extensions) == 0
    assert _ids(x3p) == [VENDOR]
    with pytest.raises(X3pFormatError, match="single vendor"):
        x3p.extensions.add("other.example", "b.dat", b"2")
    x3p.extensions.clear()
    assert _ids(x3p) == []
    x3p.extensions.add("other.example", "b.dat", b"2")
    assert _ids(x3p) == ["other.example"]


def test_a_file_that_is_known_by_its_full_id_needs_a_vendor_first() -> None:
    x3p = _file(BASE)
    with pytest.raises(X3pFormatError, match=r"add\(vendor, path, content\)"):
        x3p.extensions[f"{VENDOR}/a.dat"] = b"1"
    assert len(x3p.extensions) == 0
    x3p.extensions.add(VENDOR, "a.dat", b"1")
    x3p.extensions[f"{VENDOR}/b.dat"] = b"2"
    x3p.extensions[f"{VENDOR}/notes/c.dat"] = b"3"
    assert dict(x3p.extensions) == _keys({"a.dat": b"1", "b.dat": b"2", "notes/c.dat": b"3"})
    with pytest.raises(X3pFormatError, match="does not lie below the vendor"):
        x3p.extensions["other.example/d.dat"] = b"4"
    with pytest.raises(X3pFormatError, match="does not lie below the vendor"):
        x3p.extensions[f"{VENDOR}x/d.dat"] = b"4"


def test_the_id_around_whitespace_is_written_without_it() -> None:
    x3p = _file(BASE)
    x3p.extensions.add(f"  {VENDOR}\n", "a.dat", b"1")
    assert (
        f"<VendorSpecificID>{VENDOR}</VendorSpecificID>" in unzip(x3p.dumps())["main.xml"].decode()
    )


def test_the_id_cannot_hold_characters_xml_cannot() -> None:
    x3p = _file(BASE)
    x3p.extensions.add("a\x00b", "a.dat", b"1")
    with pytest.raises(X3pFormatError):
        x3p.dumps()


def test_a_vendor_cannot_be_empty() -> None:
    for vendor in ("", "  ", "/"):
        with pytest.raises(X3pFormatError, match="empty"):
            _file(BASE).extensions.add(vendor, "a.dat", b"1")


# --- adding files -------------------------------------------------------------------------------


@pytest.mark.parametrize("revision", list(Revision))
def test_the_path_is_below_the_vendor_in_every_revision(revision: Revision) -> None:
    x3p = _file(revision)
    x3p.extensions.add("http://www.vendor.com", "mypath/a.xml", b"<a/>")
    x3p.extensions.add("http://www.vendor.com/", "b.xml", b"<b/>")
    x3p.extensions.add("http://www.vendor.com", "sub\\c.xml", b"<c/>")
    assert dict(x3p.extensions) == {
        "http://www.vendor.com/mypath/a.xml": b"<a/>",
        "http://www.vendor.com/b.xml": b"<b/>",
        "http://www.vendor.com/sub/c.xml": b"<c/>",
    }
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_amd_1_takes_files_of_several_vendors() -> None:
    x3p = _file(AMENDED)
    x3p.extensions.add("http://www.vendor.com", "a.xml", b"<a/>")
    x3p.extensions.add("http://other.example", "b.xml", b"<b/>")
    assert sorted(x3p.extensions) == [
        "http://other.example/b.xml",
        "http://www.vendor.com/a.xml",
    ]
    assert _ids(x3p) == [
        "http://www.vendor.com/a.xml",
        "http://other.example/b.xml",
    ]


@pytest.mark.parametrize("revision", list(Revision))
@pytest.mark.parametrize("path", ["", "/a.dat", "../a.dat", "a/../b.dat", "http://x/a.dat"])
def test_a_path_must_name_a_file_inside_of_the_container(revision: Revision, path: str) -> None:
    x3p = _file(revision)
    with pytest.raises(X3pFormatError):
        x3p.extensions.add("http://www.vendor.com", path, b"1")
    assert len(x3p.extensions) == 0
    assert _ids(x3p) == []


@pytest.mark.parametrize("name", ["main.xml", "md5checksum.hex", "bindata/data.bin"])
def test_a_file_cannot_take_the_place_of_a_file_of_the_format(name: str) -> None:
    with pytest.raises(X3pFormatError, match="format"):
        _file(BASE).extensions.add(VENDOR, name, b"1")


def test_a_file_cannot_also_be_a_directory() -> None:
    x3p = _file(BASE)
    x3p.extensions.add(VENDOR, "a/b.dat", b"1")
    with pytest.raises(X3pFormatError, match="directory"):
        x3p.extensions.add(VENDOR, "a", b"2")
    with pytest.raises(X3pFormatError, match="directory"):
        x3p.extensions.add(VENDOR, "a/b.dat/c", b"2")
    assert dict(x3p.extensions) == _keys({"a/b.dat": b"1"})


def test_a_name_like_a_file_of_the_format_is_fine_if_it_differs() -> None:
    _file(BASE).extensions.add(VENDOR, "sub/main.xml", b"1")


def test_a_file_is_replaced_by_adding_it_again() -> None:
    x3p = _file(BASE)
    x3p.extensions.add(VENDOR, "a.dat", b"1")
    x3p.extensions.add(VENDOR, "a.dat", b"2")
    assert dict(x3p.extensions) == _keys({"a.dat": b"2"})


def test_the_content_may_be_a_path_or_a_stream(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"from a path")
    extensions = _file(BASE).extensions
    extensions.add(VENDOR, "a.dat", source)
    extensions.add(VENDOR, "b.dat", io.BytesIO(b"from a stream"))
    extensions.add(VENDOR, "c.dat", memoryview(b"view"))
    extensions.add(VENDOR, "d.dat", str(source))  # a str is a path, never data
    assert dict(extensions) == _keys(
        {
            "a.dat": b"from a path",
            "b.dat": b"from a stream",
            "c.dat": b"view",
            "d.dat": b"from a path",
        }
    )


def test_a_missing_source_changes_nothing_and_fixes_no_vendor(tmp_path: Path) -> None:
    x3p = _file(BASE)
    with pytest.raises(OSError, match="missing"):
        x3p.extensions.add(VENDOR, "a.dat", tmp_path / "missing")
    assert len(x3p.extensions) == 0
    assert _ids(x3p) == []


def test_a_file_is_extracted_to_a_path_or_a_stream(tmp_path: Path) -> None:
    extensions = _file(BASE).extensions
    extensions.add(VENDOR, "a.dat", b"payload")
    extensions.extract(f"{VENDOR}/a.dat", tmp_path / "out.bin")
    assert (tmp_path / "out.bin").read_bytes() == b"payload"
    buffer = io.BytesIO()
    extensions.extract(f"{VENDOR}/a.dat", buffer)
    assert buffer.getvalue() == b"payload"
    with pytest.raises(KeyError):
        extensions.extract(f"{VENDOR}/b.dat", tmp_path / "x.bin")


def test_a_vendor_or_a_directory_is_a_view_in_the_standard_too() -> None:
    vendor = "http://www.example.org/format"
    x3p = _file(BASE)
    x3p.extensions.add(vendor, "a.dat", b"1")
    x3p.extensions.add(vendor, "notes/b.txt", b"2")
    assert list(x3p.extensions.below("www.example.org/format/notes")) == [f"{vendor}/notes/b.txt"]
    assert len(x3p.extensions.below("www.example.org")) == 2
    view = x3p.extensions.below("www.example.org/format/notes")
    view.add(vendor, "notes/c.txt", b"3")
    with pytest.raises(X3pFormatError, match="lie below"):
        view.add(vendor, "d.dat", b"4")
    view.clear()
    assert list(x3p.extensions) == [f"{vendor}/a.dat"]
    assert _ids(x3p) == [vendor]  # clearing a view does not forget the vendor


def test_the_extensions_are_compared_with_their_revision_and_vendor() -> None:
    first, second = _file(BASE).extensions, _file(BASE).extensions
    first.add(VENDOR, "a.dat", b"1")
    second.add(VENDOR + "/", "a.dat", b"1")
    assert first == second
    other = _file(BASE).extensions
    other.add("other.example", "a.dat", b"1")
    assert first != other
    second[f"{VENDOR}/a.dat"] = b"2"
    assert first != second
    assert first == {f"{VENDOR}/a.dat": b"1"}
    with pytest.raises(TypeError):
        hash(first)


def test_a_vendor_without_files_is_part_of_the_comparison() -> None:
    assert _read_base([VENDOR], {}).extensions != _read_base([], {}).extensions
    assert _read_base([VENDOR], {}).extensions == _read_base([VENDOR + "/"], {}).extensions
    assert VendorExtensions(AMENDED) != VendorExtensions(BASE)


def test_a_copy_does_not_share_the_files_or_the_vendor() -> None:
    original = _read_base([VENDOR], FILES)
    copy = original.with_metadata(comment="changed")
    copy.extensions.add(VENDOR, "new.dat", b"1")
    copy.extensions.clear()
    assert dict(original.extensions) == _keys(FILES)
    assert _ids(original) == [VENDOR]
    assert _ids(copy) == []


# --- converting between the revisions ------------------------------------------------------------


def test_the_same_revision_gives_an_independent_copy() -> None:
    original = _read_base([VENDOR], FILES)
    copy = original.with_revision(BASE)
    assert copy == original
    copy.extensions.add(VENDOR, "new.dat", b"1")
    assert f"{VENDOR}/new.dat" not in original.extensions


def test_a_conversion_changes_the_calibration_date_only() -> None:
    original = array_file(np.arange(6.0).reshape(2, 3), x_scale=1e-6, y_scale=2e-6)
    original = original.with_metadata(creator="Jane Doe")
    converted = original.with_revision(BASE)
    assert converted.revision is BASE
    assert original.revision is AMENDED
    assert np.array_equal(stacked(converted), stacked(original))
    assert converted.metadata is not None
    assert original.metadata is not None
    # The schema of ISO5436_2000 requires a calibration date, which the creation date gives.
    assert converted.metadata.calibration_date == original.metadata.date
    assert replace(converted.metadata, calibration_date=None) == original.metadata
    assert original.metadata.calibration_date is None
    assert original.with_revision(AMENDED).metadata == original.metadata
    assert converted.header.x == original.header.x
    assert converted.storage is original.storage
    assert not stacked(converted).flags.writeable  # the arrays are shared as read-only views


def test_the_files_of_the_standard_are_the_extensions_of_amd_1_as_they_are() -> None:
    x3p = _read_base([VENDOR], FILES).with_revision(AMENDED)
    assert x3p.revision is AMENDED
    assert x3p.extensions._revision is AMENDED
    assert dict(x3p.extensions) == _keys(FILES)
    assert VENDOR not in x3p.extensions


def test_the_converted_file_is_written_in_the_layout_of_amd_1() -> None:
    x3p = _read_base([VENDOR], FILES).with_revision(AMENDED)
    blob = x3p.dumps()
    members = unzip(blob)
    # With backslashes, as the standard writes it, on every platform; the zip reader of Windows
    # reads them as slashes, so look at the bytes.
    assert b"nanofocus\\de\\intensity.dat" in blob
    assert not {"intensity.dat", "color.dat", "metadata.npsx"} & members.keys()
    text = members["main.xml"].decode()
    assert text.count("<VendorSpecificID>") == 3
    assert f"<VendorSpecificID>{VENDOR}</VendorSpecificID>" not in text
    SCHEMAS[AMENDED].validate(text)
    assert X3pFile.loads(blob) == x3p


def test_a_full_uri_gives_the_files_ids_below_it() -> None:
    vendor = "http://www.example.org/examplefileformat"
    x3p = _read_base([vendor], {"notes_of_sample.txt": b"text"}).with_revision(AMENDED)
    assert dict(x3p.extensions) == {f"{vendor}/notes_of_sample.txt": b"text"}
    assert b"www\\example\\org\\examplefileformat\\notes_of_sample.txt" in x3p.dumps()


def test_a_trailing_slash_of_the_id_is_not_doubled() -> None:
    x3p = _read_base([VENDOR + "/"], {"a.dat": b"1"}).with_revision(AMENDED)
    assert list(x3p.extensions) == [f"{VENDOR}/a.dat"]


def test_a_vendor_without_files_stays_an_id_that_names_no_file() -> None:
    x3p = _read_base([VENDOR], {}).with_revision(AMENDED)
    assert dict(x3p.extensions) == {VENDOR: None}


def test_a_file_without_an_id_converts_to_no_extensions() -> None:
    assert len(_read_base([], {"a.dat": b"1"}).with_revision(AMENDED).extensions) == 0


def test_a_file_that_cannot_get_an_id_of_its_own_is_an_error() -> None:
    x3p = _read_base([VENDOR], {"what?.dat": b"1"})
    with pytest.raises(X3pFormatError, match="what"):
        x3p.with_revision(AMENDED)


def test_the_files_of_a_conversion_are_independent() -> None:
    x3p = _read_base([VENDOR], FILES)
    converted = x3p.with_revision(AMENDED)
    converted.extensions[f"{VENDOR}/new.dat"] = b"1"
    assert f"{VENDOR}/new.dat" not in x3p.extensions


VENDOR_URL = "http://www.vendor.com"
AMENDED_IDS = {
    f"{VENDOR_URL}/mypath/a.xml": b"<a/>",
    f"{VENDOR_URL}/b.txt": b"text",
}


def _amended_with(ids: Mapping[str, bytes | None]) -> X3pFile:
    x3p = _file(AMENDED)
    for uri, content in ids.items():
        if content is None:
            x3p.extensions._announce(uri, None, None)
        else:
            x3p.extensions[uri] = content
    return x3p


def _base_with(files: Mapping[str, bytes]) -> X3pFile:
    x3p = _file(BASE)
    for path, content in files.items():
        x3p.extensions.add(VENDOR_URL, path, content)
    return x3p


def test_the_files_of_the_standard_become_ids_of_amd_1() -> None:
    base = _base_with({"mypath/a.xml": b"<a/>", "b.txt": b"text"})
    blob = base.dumps()
    members = unzip(blob)
    assert members["mypath/a.xml"] == b"<a/>"
    assert members["b.txt"] == b"text"
    text = members["main.xml"].decode()
    assert text.count("<VendorSpecificID>") == 1
    assert f"<VendorSpecificID>{VENDOR_URL}</VendorSpecificID>" in text
    amended = X3pFile.loads(blob).with_revision(AMENDED)
    assert dict(amended.extensions) == AMENDED_IDS


def test_the_extensions_cannot_be_converted_to_the_standard() -> None:
    x3p = _amended_with(AMENDED_IDS)
    with pytest.raises(X3pFormatError, match="cannot be converted"):
        x3p.with_revision(BASE)


def test_an_id_that_names_no_file_counts_as_an_extension() -> None:
    x3p = _amended_with({VENDOR_URL: None})
    with pytest.raises(X3pFormatError, match="cannot be converted"):
        x3p.with_revision(BASE)


def test_nothing_is_dropped_without_extensions() -> None:
    converted = _file(AMENDED).with_revision(BASE)
    assert converted.revision is BASE
    assert _ids(converted) == []
    assert len(converted.extensions) == 0


def test_the_extensions_can_be_dropped_instead() -> None:
    x3p = _amended_with({**AMENDED_IDS, "http://other.example/c.bin": b"c"})
    converted = x3p.with_revision(BASE, drop_extensions=True)
    assert converted.revision is BASE
    assert len(converted.extensions) == 0
    assert _ids(converted) == []
    assert len(x3p.extensions) == 3  # this file is as it was


def test_converting_to_amd_1_needs_no_further_information() -> None:
    x3p = _read_base([VENDOR], FILES)
    assert dict(x3p.with_revision(AMENDED).extensions) == _keys(FILES)


# --- the command line ---------------------------------------------------------------------------


def _write_base(path: Path) -> Path:
    path.write_bytes(_read_base([VENDOR], FILES).dumps())
    return path


def test_info_shows_the_revision_and_the_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["info", str(_write_base(tmp_path / "a.x3p"))]) == 0
    output = capsys.readouterr().out
    assert "Revision         = ISO5436 - 2000" in output
    assert f"VendorSpecificID = {VENDOR}" in output
    assert "VendorFile       = intensity.dat" in output


def test_info_shows_the_revision_of_amd_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "a.x3p"
    path.write_bytes(build(DataStorage.BINARY))
    assert main(["info", str(path)]) == 0
    assert "Revision        = ISO25178-72:2017/DAM1" in capsys.readouterr().out


def test_convert_changes_the_revision_and_the_extensions(tmp_path: Path) -> None:
    source = _write_base(tmp_path / "a.x3p")
    target = tmp_path / "b.x3p"
    assert main(["convert", "--revision", "ISO25178_72_2017_DAM1", str(source), str(target)]) == 0
    converted = x3pio.read(target)
    assert converted.revision is AMENDED
    assert dict(converted.extensions) == _keys(FILES)


def test_convert_to_the_standard_fails_with_extensions_without_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "a.x3p"
    _amended_with(AMENDED_IDS).save(source)
    target = tmp_path / "b.x3p"
    assert main(["convert", "-r", "ISO5436_2000", str(source), str(target)]) == 1
    assert "cannot be converted" in capsys.readouterr().err
    assert not target.exists()


def test_convert_with_fix_drops_the_extensions(tmp_path: Path) -> None:
    source = tmp_path / "a.x3p"
    _amended_with(AMENDED_IDS).save(source)
    target = tmp_path / "b.x3p"
    args = ["convert", "-r", "ISO5436_2000", "--fix", str(source), str(target)]
    assert main(args) == 0
    assert len(x3pio.read(target).extensions) == 0


def test_the_vendor_ids_are_those_that_a_file_gets() -> None:
    amended = _file(AMENDED)
    amended.extensions.add("http://www.vendor.com", "a.xml", b"<a/>")
    amended.extensions["http://other.example/b.bin"] = b"1"
    assert amended.extensions.vendor_ids == list(amended.extensions)
    base = _file(BASE)
    assert base.extensions.vendor_ids == []
    base.extensions.add("http://www.vendor.com", "a.xml", b"<a/>")
    base.extensions.add("http://www.vendor.com", "b.xml", b"<b/>")
    assert base.extensions.vendor_ids == ["http://www.vendor.com"]  # one for the whole extension
