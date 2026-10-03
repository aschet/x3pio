# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""The MD5 checksums of a file are verified when it is read."""

from __future__ import annotations

import io

import pytest

import x3pio
from helpers import build, edit_main, replace, rezip, stacked, sub, unzip
from x3pio import DataStorage, DataType, X3pChecksumError, X3pFile, X3pFormatError
from x3pio.codec.archive import format_checksum_file, md5_hex


def _changed_main() -> bytes:
    return edit_main(
        build(), replace("<Offset>0.0</Offset>", "<Offset>0</Offset>"), refresh_checksum=False
    )


def _changed(blob: bytes, name: str) -> bytes:
    members = unzip(blob)
    content = bytearray(members[name])
    content[0] ^= 1
    members[name] = bytes(content)
    return rezip(members)


def test_a_file_written_by_x3pio_verifies() -> None:
    for storage in DataStorage:
        assert stacked(X3pFile.loads(build(storage))).shape == (1, 2, 3)
    assert stacked(X3pFile.loads(build(DataStorage.BINARY, z_type=DataType.INT16))).shape == (
        1,
        2,
        3,
    )


def test_the_error_is_a_format_error() -> None:
    assert issubclass(X3pChecksumError, X3pFormatError)


def test_a_changed_main_xml_is_refused() -> None:
    blob = _changed_main()
    with pytest.raises(X3pChecksumError, match=r"main\.xml does not match"):
        X3pFile.loads(blob)
    assert stacked(X3pFile.loads(blob, verify=False)).shape == (1, 2, 3)


def test_a_changed_data_file_is_refused() -> None:
    blob = _changed(build(DataStorage.BINARY), "bindata/data.bin")
    with pytest.raises(X3pChecksumError, match=r"data\.bin"):
        X3pFile.loads(blob)
    assert stacked(X3pFile.loads(blob, verify=False)).shape == (1, 2, 3)


def test_a_changed_validity_file_is_refused() -> None:
    blob = _changed(build(DataStorage.BINARY, z_type=DataType.INT16), "bindata/valid.bin")
    with pytest.raises(X3pChecksumError, match=r"valid\.bin"):
        X3pFile.loads(blob)
    assert stacked(X3pFile.loads(blob, verify=False)).shape == (1, 2, 3)


def test_read_and_open_verify_too(tmp_path: pytest.TempPathFactory) -> None:
    blob = _changed_main()
    with pytest.raises(X3pChecksumError):
        x3pio.read(io.BytesIO(blob))
    with pytest.raises(X3pChecksumError):
        X3pFile.open(io.BytesIO(blob))
    assert stacked(x3pio.read(io.BytesIO(blob), verify=False)).shape == (1, 2, 3)
    assert stacked(X3pFile.open(io.BytesIO(blob), verify=False)).shape == (1, 2, 3)


def test_a_missing_checksum_file_leaves_nothing_to_check() -> None:
    members = unzip(_changed_main())
    del members["md5checksum.hex"]
    assert stacked(X3pFile.loads(rezip(members))).shape == (1, 2, 3)


@pytest.mark.parametrize("content", [b"", b"garbage", b"abc *main.xml", b"z" * 32])
def test_a_checksum_file_without_a_digest_leaves_nothing_to_check(content: bytes) -> None:
    members = unzip(_changed_main())
    members["md5checksum.hex"] = content
    assert stacked(X3pFile.loads(rezip(members))).shape == (1, 2, 3)


@pytest.mark.parametrize(
    "layout", ["{digest}", "{digest} *main.xml", "{DIGEST}\r\n", "﻿{digest}  main.xml\n"]
)
def test_the_layouts_of_a_checksum_file_are_accepted(layout: str) -> None:
    members = unzip(build())
    digest = members["md5checksum.hex"].split()[0].decode("ascii")
    members["md5checksum.hex"] = layout.format(digest=digest, DIGEST=digest.upper()).encode()
    assert stacked(X3pFile.loads(rezip(members))).shape == (1, 2, 3)


@pytest.mark.parametrize("layout", ["{digest}", "{DIGEST}\r\n"])
def test_a_wrong_digest_is_refused_in_any_layout(layout: str) -> None:
    members = unzip(build())
    members["md5checksum.hex"] = layout.format(digest="0" * 32, DIGEST="0" * 32).encode()
    with pytest.raises(X3pChecksumError):
        X3pFile.loads(rezip(members))


def test_a_checksum_file_may_have_another_name() -> None:
    members = unzip(edit_main(build(), replace("md5checksum.hex", "sum.md5")))
    members["sum.md5"] = members.pop("md5checksum.hex")
    assert stacked(X3pFile.loads(rezip(members))).shape == (1, 2, 3)
    members["sum.md5"] = b"0" * 32
    with pytest.raises(X3pChecksumError):
        X3pFile.loads(rezip(members))


def test_a_checksum_file_that_is_not_in_the_container_is_not_checked() -> None:
    blob = edit_main(_changed_main(), replace("md5checksum.hex", "../md5checksum.hex"))
    assert stacked(X3pFile.loads(blob)).shape == (1, 2, 3)


def test_a_data_checksum_that_is_not_a_digest_is_not_checked() -> None:
    blob = edit_main(
        build(DataStorage.BINARY),
        sub(r"<MD5ChecksumPointData>[^<]*<", "<MD5ChecksumPointData>xyz<"),
    )
    assert stacked(X3pFile.loads(_changed(blob, "bindata/data.bin"))).shape == (1, 2, 3)


def test_a_missing_data_checksum_is_not_checked() -> None:
    blob = edit_main(
        build(DataStorage.BINARY, z_type=DataType.INT16),
        lambda text: text.replace("MD5ChecksumPointData", "Nothing").replace(
            "MD5ChecksumValidPoints", "Nothing"
        ),
    )
    blob = _changed(_changed(blob, "bindata/data.bin"), "bindata/valid.bin")
    assert stacked(X3pFile.loads(blob)).shape == (1, 2, 3)


def test_the_digests_are_compared_without_regard_to_case() -> None:
    members = unzip(build(DataStorage.BINARY))
    text = members["main.xml"].decode()
    digest = text.split("<MD5ChecksumPointData>")[1].split("<")[0]
    members["main.xml"] = text.replace(digest, digest.upper()).encode()
    members["md5checksum.hex"] = format_checksum_file(md5_hex(members["main.xml"]))
    assert stacked(X3pFile.loads(rezip(members))).shape == (1, 2, 3)
