# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import zipfile
import zlib

import numpy as np
import pytest

from helpers import rezip
from x3pio import X3pFormatError
from x3pio.codec.archive import (
    Container,
    PackedMember,
    format_checksum_file,
    md5_hex,
    pack_validity,
    unpack_validity,
    write_container,
)
from x3pio.model.naming import is_reserved, local_member_path, vendor_extension_path

MD5_OF_EMPTY = "d41d8cd98f00b204e9800998ecf8427e"


def test_md5_hex() -> None:
    assert md5_hex(b"") == MD5_OF_EMPTY


def test_checksum_file_is_formatted_like_md5sum() -> None:
    assert format_checksum_file(MD5_OF_EMPTY) == f"{MD5_OF_EMPTY} *main.xml\n".encode("ascii")


def test_validity_bits_start_at_the_least_significant_bit() -> None:
    # The sample file of the standard marks the 8th of 16 points invalid as 7f ff.
    valid = np.ones(16, dtype=bool)
    valid[7] = False
    assert pack_validity(valid) == bytes([0x7F, 0xFF])
    assert np.array_equal(unpack_validity(bytes([0x7F, 0xFF]), 16), valid)


def test_validity_bit_index_matches_the_standard_table() -> None:
    # Bit j is bit j % 8 of byte j // 8 (the example of Amd 1:2020).
    for index in range(17):
        valid = np.zeros(17, dtype=bool)
        valid[index] = True
        packed = pack_validity(valid)
        assert packed[index // 8] == 1 << (index % 8)


def test_validity_padding_is_ignored() -> None:
    assert unpack_validity(bytes([0xFF, 0xFF]), 10).tolist() == [True] * 10


@pytest.mark.parametrize(
    ("link", "expected"),
    [
        ("bindata/data.bin", "bindata/data.bin"),
        ("./bindata\\data.bin", "bindata/data.bin"),
        ("  main.xml ", "main.xml"),
        ("a//b", "a/b"),
    ],
)
def test_local_member_path_normalizes(link: str, expected: str) -> None:
    assert local_member_path(link) == expected


@pytest.mark.parametrize(
    "link",
    [
        "",
        "/etc/passwd",
        "\\\\server\\share\\data.bin",
        "http://example.org/data.bin",
        "file:///data.bin",
        "C:\\data.bin",
        "../data.bin",
        "bindata/../../data.bin",
    ],
)
def test_local_member_path_rejects_non_local_links(link: str) -> None:
    with pytest.raises(X3pFormatError, match="does not point to a member"):
        local_member_path(link)


def test_reserved_names() -> None:
    assert is_reserved("bindata\\data.bin")
    assert is_reserved("./main.xml")
    assert not is_reserved("bindata/other.bin")


def test_vendor_extension_path_follows_the_standard_example() -> None:
    uri = "http://www.vendor.com/mypath/myelements.xml"
    assert vendor_extension_path(uri) == "www\\vendor\\com\\mypath\\myelements.xml"


@pytest.mark.parametrize(
    "uri", ["nanofocus.de", "http://www.vendor.com", "http://a.com/../x", "x//y"]
)
def test_vendor_extension_path_needs_a_path_to_a_file(uri: str) -> None:
    with pytest.raises(X3pFormatError, match="VendorSpecificID"):
        vendor_extension_path(uri)


def test_write_container_is_deterministic_and_keeps_names_exactly() -> None:
    members = [("main.xml", b"<a/>"), ("a\\b.dat", b"x"), ("c/d.dat", b"y")]
    blob = write_container(members)
    assert blob == write_container(members)
    # The name is in the headers as given, whatever the platform does when reading it back.
    assert b"a\\b.dat" in blob
    assert b"c/d.dat" in blob
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}


def test_container_rejects_a_file_that_is_not_a_zip() -> None:
    with pytest.raises(X3pFormatError, match="not a zip archive"):
        Container(b"not a zip")


def test_container_without_main_xml_is_rejected() -> None:
    with pytest.raises(X3pFormatError, match=r"no main\.xml"):
        Container(rezip({"other.xml": b""}))


def test_container_with_several_nested_main_xml_is_rejected() -> None:
    with pytest.raises(X3pFormatError, match="several folders"):
        Container(rezip({"a/main.xml": b"", "b/main.xml": b""}))


def test_container_ignores_directory_entries() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bindata/", b"")
        archive.writestr("main.xml", b"<a/>")
    assert Container(buffer.getvalue()).names == ["main.xml"]


def test_container_drops_junk_members() -> None:
    blob = rezip(
        {
            "main.xml": b"<a/>",
            "__MACOSX/._main.xml": b"",
            "bindata/.DS_Store": b"",
            "Thumbs.db": b"",
            "bindata/._d.bin": b"",
        }
    )
    assert Container(blob).names == ["main.xml"]


@pytest.mark.parametrize(
    "junk",
    [
        "Thumbs.db",
        "thumbs.db",
        "THUMBS.DB",
        "bindata/Thumbs.db",
        "desktop.ini",
        "Desktop.ini",
        "sub/DESKTOP.INI",
        ".DS_Store",
        ".ds_store",
        "sub/dir/.DS_Store",
        "__MACOSX/a.dat",
        "__MACOSX/sub/deeper/b.dat",
        "__macosx/a.dat",
        "sample/__MACOSX/a.dat",
        "sample/bindata/__MACOSX/sub/a.dat",
        "._a.dat",
        "bindata/._a.dat",
    ],
)
def test_container_drops_what_an_operating_system_adds(junk: str) -> None:
    assert Container(rezip({"main.xml": b"<a/>", junk: b"x", "a.dat": b"1"})).names == [
        "main.xml",
        "a.dat",
    ]


@pytest.mark.parametrize(
    "name",
    ["__MACOSX", "sub/__MACOSX", "__MACOSXa/x.dat", "x__MACOSX/a.dat", "a._b.dat", ".directory",
     "thumbs.db.bak", "Thumbs.dbx", "my_desktop.ini", "a.dat~"],
)  # fmt: skip
def test_container_keeps_files_that_only_look_alike(name: str) -> None:
    assert name in Container(rezip({"main.xml": b"<a/>", name: b"x"})).names


def test_container_finds_content_nested_in_one_folder() -> None:
    container = Container(rezip({"sample/main.xml": b"<a/>", "sample/bindata/d.bin": b"1"}))
    assert sorted(container.names) == ["bindata/d.bin", "main.xml"]
    assert container.read("bindata/d.bin") == b"1"
    assert container.size("bindata/d.bin") == 1


def test_container_ignores_members_beside_the_nested_folder() -> None:
    container = Container(rezip({"sample/main.xml": b"<a/>", "readme.txt": b"hi"}))
    assert container.names == ["main.xml"]


def test_container_accepts_backslash_separators() -> None:
    container = Container(rezip({"main.xml": b"<a/>", "a\\b.dat": b"x"}))
    assert container.names == ["main.xml", "a/b.dat"]
    assert container.has("a/b.dat")
    assert container.read("a/b.dat") == b"x"


def test_container_reports_a_damaged_member() -> None:
    blob = bytearray(rezip({"main.xml": b"<a/>" * 100}))
    blob[40] ^= 0xFF  # inside the compressed data
    container = Container(bytes(blob))
    with pytest.raises(X3pFormatError, match="Cannot read"):
        container.read("main.xml")


def _packed(content: bytes, *, deflated: bool = True, **override: int) -> PackedMember:
    raw = zlib.compress(content)[2:-4] if deflated else content
    fields = {"size": len(content), "crc": zlib.crc32(content), **override}
    return PackedMember("name", raw, deflated=deflated, **fields)


@pytest.mark.parametrize("deflated", [True, False])
def test_a_packed_member_returns_its_content_every_time(deflated: bool) -> None:
    member = _packed(b"abc" * 100, deflated=deflated)
    assert member.read() == b"abc" * 100
    assert member.read() == b"abc" * 100


@pytest.mark.parametrize("deflated", [True, False])
@pytest.mark.parametrize(
    "override", [{"size": 5}, {"size": 500}, {"crc": 1}], ids=["small", "large", "crc"]
)
def test_a_packed_member_that_does_not_match_its_declaration_is_damaged(
    deflated: bool, override: dict[str, int]
) -> None:
    with pytest.raises(X3pFormatError, match="name"):
        _packed(b"abc" * 100, deflated=deflated, **override).read()


def test_a_packed_member_with_invalid_deflate_data_is_damaged() -> None:
    member = PackedMember("name", b"\xff\xff\xff", deflated=True, size=3, crc=0)
    with pytest.raises(X3pFormatError, match="name"):
        member.read()


def test_the_empty_packed_member() -> None:
    assert _packed(b"").read() == b""
