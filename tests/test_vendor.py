# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import xmlschema

from helpers import array_file, build, edit_main, rezip, stacked, unzip
from x3pio import DataStorage, Surface, VendorExtensions, X3pFile, X3pFormatError
from x3pio.codec.archive import PackedMember
from x3pio.model.naming import split_uri, vendor_extension_path

OFFICIAL = xmlschema.XMLSchema(Path(__file__).parent / "xsd" / "iso25178-72-fdam1.xsd")

URI = "http://www.vendor.com/mypath/myelements.xml"
#: The name the standard gives the file of URI in the container, and the form with slashes.
SPEC_NAME = "www\\vendor\\com\\mypath\\myelements.xml"
SLASH_NAME = "www/vendor/com/mypath/myelements.xml"


def _file() -> X3pFile:
    return array_file(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)


def _ext(x3p: X3pFile) -> VendorExtensions:
    """Return the extensions of a file in the revision of Amd 1:2020."""
    assert isinstance(x3p.extensions, VendorExtensions)
    return x3p.extensions


def _with_id(blob: bytes, *uris: str) -> bytes:
    ids = "".join(f"<VendorSpecificID>{uri}</VendorSpecificID>" for uri in uris)
    return edit_main(blob, lambda text: text.replace("</p:ISO5436_2>", ids + "</p:ISO5436_2>"))


def _read_with(uris: list[str], members: dict[str, bytes]) -> X3pFile:
    files = unzip(_with_id(build(), *uris) if uris else build())
    files.update(members)
    return X3pFile.loads(rezip(files))


# --- the name in the container ------------------------------------------------------------------


def test_the_name_in_the_container_is_derived_from_the_uri() -> None:
    # The example of Amd 1:2020 for the name of the file of an ID.
    assert vendor_extension_path(URI) == SPEC_NAME


@pytest.mark.parametrize(
    "uri",
    [
        "https://www.vendor.com/mypath/myelements.xml",
        "www.vendor.com/mypath/myelements.xml",
        "ftp://www.vendor.com/mypath/myelements.xml",
    ],
)
def test_the_scheme_is_not_part_of_the_name(uri: str) -> None:
    assert vendor_extension_path(uri) == SPEC_NAME


def test_a_dot_in_the_file_name_is_kept() -> None:
    assert vendor_extension_path("http://a.b/c.d.e") == "a\\b\\c.d.e"


def test_a_uri_is_split_into_its_host_and_the_segments_of_its_path() -> None:
    assert split_uri(URI) == ("www.vendor.com", ["mypath", "myelements.xml"])
    assert split_uri("www.vendor.com/mypath/a.xml") == ("www.vendor.com", ["mypath", "a.xml"])
    assert split_uri("http://www.vendor.com") == ("www.vendor.com", [])
    assert split_uri("http://www.vendor.com/") == ("www.vendor.com", [])
    assert split_uri("nanofocus.de") == ("nanofocus.de", [])


def test_a_port_or_a_user_name_is_only_refused_when_registering() -> None:
    assert split_uri("http://u@www.vendor.com:80/a.xml") == ("u@www.vendor.com:80", ["a.xml"])
    for uri in ("http://www.vendor.com:80/a.xml", "http://u@www.vendor.com/a.xml"):
        with pytest.raises(X3pFormatError):
            split_uri(uri, registering=True)


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "http://",
        "a//b",
        "http://a.b/c/",
        "a.b/../c",
        "a.b/./c",
        ".b/c",
        "a..b/c",
        "http://a.b/c?x=1",
        "http://a.b/c#f",
        "http://a.b/c\\d",
        "http://[a.b/c",
    ],
)
def test_a_uri_that_gives_no_name_is_an_error(uri: str) -> None:
    with pytest.raises(X3pFormatError):
        split_uri(uri)


# --- registering ------------------------------------------------------------------------------


def test_the_extension_is_written_under_the_name_the_standard_gives_it() -> None:
    x3p = _file()
    x3p.extensions[URI] = b"<ext/>"

    assert x3p.extensions == {URI: b"<ext/>"}
    blob = x3p.dumps()
    # With backslashes, as the standard writes it, on every platform.
    assert SPEC_NAME.encode() in blob
    assert SLASH_NAME.encode() not in blob
    assert "<VendorSpecificID>" + URI in unzip(blob)["main.xml"].decode()


@pytest.mark.parametrize("storage", list(DataStorage))
def test_extension_roundtrips(storage: DataStorage) -> None:
    x3p = _file()
    x3p.extensions[URI] = b"<ext/>"
    again = X3pFile.loads(x3p.dumps(storage=storage))
    assert again.extensions == {URI: b"<ext/>"}
    assert again == x3p


@pytest.mark.parametrize(
    "uri",
    ["http://a.b/c.bin", "https://a.b/c.bin", "HTTP://A.B/Path/c.bin", "a.b/c.bin", "ftp://x.y/z"],
)
def test_the_id_is_registered_exactly_as_given(uri: str) -> None:
    x3p = _file()
    x3p.extensions[uri] = b"1"
    assert list(x3p.extensions) == [uri]
    assert list(X3pFile.loads(x3p.dumps()).extensions) == [uri]


def test_the_whitespace_around_an_id_is_not_part_of_it() -> None:
    x3p = _file()
    x3p.extensions["  " + URI + "\n"] = b"1"
    assert list(x3p.extensions) == [URI]


def test_registering_again_replaces_the_content() -> None:
    x3p = _file()
    x3p.extensions[URI] = b"old"
    x3p.extensions[URI] = b"new"
    assert x3p.extensions == {URI: b"new"}
    assert len(x3p.extensions) == 1


def test_several_ids_are_written_and_valid() -> None:
    x3p = _file()
    x3p.extensions[URI] = b"1"
    x3p.extensions["http://example.org/a/b.bin"] = b"2"
    assert X3pFile.loads(x3p.dumps()).extensions == x3p.extensions
    text = unzip(x3p.dumps())["main.xml"].decode()
    assert text.count("<VendorSpecificID>") == 2
    OFFICIAL.validate(text)


def test_the_content_is_copied_into_bytes() -> None:
    extensions = VendorExtensions()
    data = bytearray(b"12")
    extensions[URI] = data  # type: ignore[assignment]
    data[0] = 0
    assert extensions[URI] == b"12"
    assert type(extensions[URI]) is bytes


def test_a_file_needs_content() -> None:
    with pytest.raises(TypeError, match="needs content"):
        _file().extensions[URI] = None


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "nanofocus.de",
        "http://www.vendor.com",
        "http://www.vendor.com/",
        "http://www.vendor.com/mypath/",
        "http://a.b/../c",
        "http://a.b//c",
        "http://www.vendor.com:8080/a.xml",
        "http://user@www.vendor.com/a.xml",
        "http://www.vendor.com/a.xml?x=1",
        "http://www.vendor.com/a.xml#frag",
        "http://a.b/c\\d",
    ],
)
def test_an_id_that_names_no_file_cannot_be_registered(uri: str) -> None:
    x3p = _file()
    with pytest.raises(X3pFormatError):
        x3p.extensions[uri] = b"1"
    assert len(x3p.extensions) == 0


def test_a_file_cannot_also_be_a_directory() -> None:
    x3p = _file()
    x3p.extensions["http://a.b/c"] = b"1"
    with pytest.raises(X3pFormatError, match="both a file and a directory"):
        x3p.extensions["http://a.b/c/d"] = b"2"
    x3p.extensions["http://a.b/e/f"] = b"3"
    with pytest.raises(X3pFormatError, match="both a file and a directory"):
        x3p.extensions["http://a.b/e"] = b"4"
    assert len(x3p.extensions) == 2


@pytest.mark.parametrize("uri", ["http://bindata/data.bin", "bindata/valid.bin"])
def test_an_extension_cannot_take_the_place_of_a_file_of_the_format(uri: str) -> None:
    with pytest.raises(X3pFormatError, match="file of the format"):
        _file().extensions[uri] = b"1"


def test_a_name_like_the_one_of_a_file_of_the_format_is_fine_if_it_differs() -> None:
    x3p = _file()
    x3p.extensions["http://main.xml/x"] = b"1"  # the name main\xml\x, not main.xml
    assert X3pFile.loads(x3p.dumps()).extensions == {"http://main.xml/x": b"1"}


def test_ids_cannot_hold_characters_xml_cannot() -> None:
    x3p = _file()
    x3p.extensions["http://a.b/c\x01d"] = b"1"
    with pytest.raises(X3pFormatError, match="VendorSpecificID"):
        x3p.dumps()


# --- two IDs, one file --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("http://a.b/c.bin", "https://a.b/c.bin"),
        ("http://a.b.c/d", "http://a.b/c/d"),
        ("http://a.b/c", "a.b/c"),
    ],
)
def test_ids_that_name_the_same_file_share_it(first: str, second: str) -> None:
    x3p = _file()
    x3p.extensions[first] = b"1"
    x3p.extensions[second] = b"1"
    assert x3p.extensions == {first: b"1", second: b"1"}
    again = X3pFile.loads(x3p.dumps())
    assert again == x3p
    assert len(unzip(x3p.dumps())) == len(unzip(_file().dumps())) + 1  # one file for both IDs


def test_a_new_id_for_a_file_with_other_content_is_an_error_and_changes_nothing() -> None:
    x3p = _file()
    x3p.extensions["http://a.b/c.bin"] = b"1"
    with pytest.raises(X3pFormatError, match=r"http://a\.b/c\.bin"):
        x3p.extensions["https://a.b/c.bin"] = b"2"
    with pytest.raises(X3pFormatError, match="same file"):
        x3p.extensions.add("https://a.b", "c.bin", b"2")
    assert x3p.extensions == {"http://a.b/c.bin": b"1"}


def test_replacing_one_of_two_ids_changes_the_file_they_share() -> None:
    x3p = _file()
    x3p.extensions["http://a.b/c.bin"] = b"1"
    x3p.extensions["https://a.b/c.bin"] = b"1"
    x3p.extensions["https://a.b/c.bin"] = b"2"
    assert x3p.extensions == {"http://a.b/c.bin": b"2", "https://a.b/c.bin": b"2"}


def test_the_file_goes_with_its_last_id() -> None:
    x3p = _file()
    x3p.extensions["http://a.b/c.bin"] = b"1"
    x3p.extensions["https://a.b/c.bin"] = b"1"
    del x3p.extensions["http://a.b/c.bin"]
    assert x3p.extensions == {"https://a.b/c.bin": b"1"}
    assert any(name.startswith("a") for name in unzip(x3p.dumps()))
    del x3p.extensions["https://a.b/c.bin"]
    assert not any(name.startswith("a\\") for name in unzip(x3p.dumps()))


# --- reading ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", [SPEC_NAME, SLASH_NAME])
def test_the_file_of_an_id_is_found_whichever_separator_the_archive_uses(name: str) -> None:
    x3p = _read_with([URI], {name: b"<ext/>"})
    assert x3p.extensions == {URI: b"<ext/>"}


def test_a_file_without_an_id_is_not_guessed_to_be_an_extension() -> None:
    # The name does not say where the host name ends, so only an ID can make it an extension.
    x3p = _read_with([], {SPEC_NAME: b"<ext/>"})
    assert len(x3p.extensions) == 0
    assert len(X3pFile.loads(x3p.dumps()).extensions) == 0


def test_an_id_without_a_file_has_no_content() -> None:
    x3p = _read_with([URI], {})
    assert x3p.extensions == {URI: None}
    assert x3p.extensions[URI] is None
    assert URI in x3p.extensions
    assert X3pFile.loads(x3p.dumps()).extensions == {URI: None}


@pytest.mark.parametrize(
    "uri", ["urn:example:vendor", "http://a.b/c?x=1", "http://a.b/c\\d", "http://a.b//c"]
)
def test_an_id_that_gives_no_name_is_kept_as_written(uri: str) -> None:
    x3p = _read_with([uri.replace("&", "&amp;")], {})
    assert x3p.extensions == {uri: None}
    assert X3pFile.loads(x3p.dumps()).extensions == {uri: None}


def test_the_file_of_an_id_with_a_port_is_found() -> None:
    uri = "http://www.vendor.com:8080/a.xml"
    x3p = _read_with([uri], {"www\\vendor\\com:8080\\a.xml": b"<a/>"})
    assert x3p.extensions == {uri: b"<a/>"}
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_unknown_ids_and_their_files_are_kept_without_interpretation() -> None:
    x3p = _read_with(
        ["http://unknown.example/x/data.bin"], {"unknown\\example\\x\\data.bin": b"\x00\x01"}
    )
    assert x3p.extensions == {"http://unknown.example/x/data.bin": b"\x00\x01"}
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_empty_vendor_ids_are_dropped_on_read() -> None:
    assert len(X3pFile.loads(_with_id(build(), " ")).extensions) == 0


def test_the_ids_are_written_back_as_they_were_read() -> None:
    uri = "HTTPS://WWW.Vendor.com/mypath/myelements.xml"
    x3p = _read_with([uri], {"WWW\\Vendor\\com\\mypath\\myelements.xml": b"<ext/>"})
    assert x3p.extensions == {uri: b"<ext/>"}
    assert list(X3pFile.loads(x3p.dumps()).extensions) == [uri]


def test_ids_of_other_schemes_with_the_same_path_share_the_file() -> None:
    other = "https://www.vendor.com/mypath/myelements.xml"
    x3p = _read_with([URI, other], {SPEC_NAME: b"<ext/>"})
    assert x3p.extensions == {URI: b"<ext/>", other: b"<ext/>"}
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_ids_that_differ_in_the_case_of_the_host_name_are_different_files() -> None:
    upper, lower = "http://WWW.vendor.com/a.xml", "http://www.vendor.com/a.xml"
    x3p = _read_with(
        [upper, lower], {"WWW\\vendor\\com\\a.xml": b"upper", "www\\vendor\\com\\a.xml": b"lower"}
    )
    assert x3p.extensions == {upper: b"upper", lower: b"lower"}
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_ids_with_different_hosts_and_paths_can_name_the_same_file() -> None:
    # Both give a\b\c\d: the standard does not make the name unique.
    ids = ["http://a.b.c/d", "http://a.b/c/d"]
    x3p = _read_with(ids, {"a\\b\\c\\d": b"same"})
    assert x3p.extensions == {ids[0]: b"same", ids[1]: b"same"}
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_junk_of_other_systems_is_skipped_also_around_an_extension() -> None:
    members = {
        SLASH_NAME: b"<ext/>",
        "www/vendor/com/mypath/.DS_Store": b"junk",
        "www/vendor/com/mypath/._myelements.xml": b"junk",
        "__MACOSX/www/vendor/com/mypath/._myelements.xml": b"junk",
        "__MACOSX/._www": b"junk",
        "Thumbs.db": b"junk",
        "bindata/": b"",
    }
    x3p = _read_with([URI], members)
    assert x3p.extensions == {URI: b"<ext/>"}
    assert X3pFile.loads(x3p.dumps()) == x3p


# --- the mapping ------------------------------------------------------------------------------

A = "http://www.vendor.com/mypath/a.xml"
B = "https://www.vendor.com/mypath/sub/b.xml"
C = "http://www.vendor.com/c.xml"
D = "http://www.vendor.com.example/d.xml"
E = "http://example.org/e.bin"


def _vendors() -> X3pFile:
    x3p = _file()
    for uri, data in ((A, b"1"), (B, b"2"), (C, b"3"), (D, b"4"), (E, b"5")):
        x3p.extensions[uri] = data
    return X3pFile.loads(x3p.dumps())


def test_the_whole_structure_is_iterated_in_file_order() -> None:
    ext = _ext(_vendors())
    assert list(ext) == [A, B, C, D, E]
    assert len(ext) == 5
    assert dict(ext.items()) == {A: b"1", B: b"2", C: b"3", D: b"4", E: b"5"}
    assert list(ext.values()) == [b"1", b"2", b"3", b"4", b"5"]


def test_the_extensions_behave_like_a_dict() -> None:
    extensions = VendorExtensions()
    extensions["http://a.b/c"] = b"1"
    assert extensions == {"http://a.b/c": b"1"}
    assert extensions != {"http://a.b/c": b"2"}
    assert extensions != "text"
    assert repr(extensions) == "VendorExtensions(['http://a.b/c'])"
    assert extensions.get("http://a.b/x") is None
    assert extensions.get("http://a.b/x", b"d") == b"d"
    assert extensions.pop("http://a.b/c") == b"1"
    assert len(extensions) == 0


def test_the_mapping_cannot_be_hashed() -> None:
    with pytest.raises(TypeError):
        hash(VendorExtensions())


def test_keys_are_exact() -> None:
    extensions = VendorExtensions()
    extensions["http://a.b/c"] = b"1"
    for other in ("https://a.b/c", "a.b/c", "http://A.b/c", "http://a.b/c/", " http://a.b/c"):
        assert other not in extensions
        with pytest.raises(KeyError):
            extensions[other]
    assert "http://a.b/c" in extensions


@pytest.mark.parametrize("name", [5, None, b"http://a.b/c"])
def test_a_key_that_is_not_a_string_is_not_in_the_mapping(name: object) -> None:
    assert name not in _vendors().extensions


def test_a_missing_id_is_a_key_error() -> None:
    ext = _ext(_vendors())
    with pytest.raises(KeyError):
        ext["http://nobody.example/x"]
    with pytest.raises(KeyError):
        del ext["http://nobody.example/x"]


def test_removing_an_id_removes_its_file() -> None:
    x3p = _vendors()
    del x3p.extensions[E]
    assert E not in x3p.extensions
    assert not any(name.startswith("example") for name in unzip(x3p.dumps()))
    assert X3pFile.loads(x3p.dumps()) == x3p


# --- one vendor or one directory ----------------------------------------------------------------


def test_a_vendor_is_found_by_its_domain_name() -> None:
    ext = _ext(_vendors())
    mine = ext.below("http://www.vendor.com")
    assert list(mine) == [A, B, C]
    assert len(mine) == 3
    assert dict(mine) == {A: b"1", B: b"2", C: b"3"}
    assert list(ext.below("example.org")) == [E]


@pytest.mark.parametrize(
    "prefix",
    ["www.vendor.com", "http://www.vendor.com", "https://www.vendor.com/", "ftp://www.vendor.com"],
)
def test_the_scheme_and_a_trailing_slash_do_not_matter_for_a_prefix(prefix: str) -> None:
    assert list(_ext(_vendors()).below(prefix)) == [A, B, C]


def test_a_directory_below_a_vendor_is_a_subtree_too() -> None:
    ext = _ext(_vendors())
    assert list(ext.below("www.vendor.com/mypath")) == [A, B]
    assert list(ext.below("www.vendor.com/mypath/sub")) == [B]
    assert list(ext.below("www.vendor.com/mypath").below("www.vendor.com/mypath/sub")) == [B]
    assert list(ext.below("www.vendor.com/mypath").below("www.vendor.com/c.xml")) == []


def test_a_prefix_matches_whole_segments_of_the_exact_host_name() -> None:
    ext = _ext(_vendors())
    assert list(ext.below("www.vendor.com/myp")) == []
    assert list(ext.below("www.vendor.com/mypath/a")) == []
    assert list(ext.below("www.vendor")) == []  # a different host name
    assert list(ext.below("vendor.com")) == []
    assert list(ext.below("WWW.vendor.com")) == []  # the case is part of the name
    assert list(ext.below("www.vendor.com.example")) == [D]
    assert list(ext.below("nobody.example")) == []


def test_a_prefix_may_be_the_id_of_a_file() -> None:
    assert list(_ext(_vendors()).below(A)) == [A]


def test_an_id_that_names_only_a_domain_lies_below_it() -> None:
    x3p = _read_with(["nanofocus.de", "http://nanofocus.de/a.bin"], {"nanofocus\\de\\a.bin": b"1"})
    assert list(_ext(x3p).below("nanofocus.de")) == [
        "nanofocus.de",
        "http://nanofocus.de/a.bin",
    ]


def test_an_id_that_is_no_uri_is_only_in_the_whole_mapping() -> None:
    x3p = _read_with(["urn:example:vendor", "http://a.b//c", URI], {SPEC_NAME: b"1"})
    assert "urn:example:vendor" in x3p.extensions
    assert "http://a.b//c" in x3p.extensions
    assert "urn:example:vendor" not in _ext(x3p).below("www.vendor.com")
    assert "http://a.b//c" not in _ext(x3p).below("a.b")
    assert list(_ext(x3p).below("www.vendor.com")) == [URI]


@pytest.mark.parametrize("prefix", ["", "http://", "a//b", "a.b/../c", "http://a.b/c?x"])
def test_an_invalid_prefix_is_an_error(prefix: str) -> None:
    with pytest.raises(X3pFormatError):
        _ext(_vendors()).below(prefix)


def test_a_subtree_reads_replaces_and_removes_in_the_file() -> None:
    x3p = _vendors()
    mine = _ext(x3p).below("www.vendor.com")
    assert mine[C] == b"3"
    assert C in mine
    assert E not in mine
    with pytest.raises(KeyError):
        mine[E]
    with pytest.raises(KeyError):
        del mine[E]
    mine[C] = b"changed"
    assert x3p.extensions[C] == b"changed"
    del mine[C]
    assert C not in x3p.extensions
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_a_subtree_is_removed_with_clear() -> None:
    x3p = _vendors()
    _ext(x3p).below("www.vendor.com/mypath").clear()
    assert list(x3p.extensions) == [C, D, E]
    _ext(x3p).below("www.vendor.com").clear()
    assert list(x3p.extensions) == [D, E]
    x3p.extensions.clear()
    assert len(x3p.extensions) == 0
    assert not any(name.startswith(("www", "example")) for name in unzip(x3p.dumps()))


def test_a_new_id_has_to_lie_below_the_prefix_of_a_view() -> None:
    x3p = _vendors()
    mine = _ext(x3p).below("www.vendor.com")
    mine["http://www.vendor.com/new.xml"] = b"6"
    mine.add("https://www.vendor.com", "other.xml", b"7")
    assert x3p.extensions["http://www.vendor.com/new.xml"] == b"6"
    with pytest.raises(X3pFormatError, match="lie below"):
        mine["http://example.org/new.xml"] = b"8"
    with pytest.raises(X3pFormatError, match="lie below"):
        mine.add("http://www.vendor.com.example", "new.xml", b"8")
    assert "http://example.org/new.xml" not in x3p.extensions


def test_the_result_of_a_dict_call_is_a_copy() -> None:
    x3p = _vendors()
    files = dict(_ext(x3p).below("www.vendor.com"))
    files[C] = b"changed"
    assert x3p.extensions[C] == b"3"


# --- extracting -------------------------------------------------------------------------------


def test_a_file_is_extracted_to_a_path_or_a_stream(tmp_path: Path) -> None:
    ext = _ext(_vendors())
    ext.extract(A, tmp_path / "a.xml")
    ext.extract(B, str(tmp_path / "b.xml"))
    assert (tmp_path / "a.xml").read_bytes() == b"1"
    assert (tmp_path / "b.xml").read_bytes() == b"2"
    stream = io.BytesIO()
    ext.extract(C, stream)
    assert stream.getvalue() == b"3"
    assert not stream.closed


def test_extracting_what_is_not_there_is_an_error(tmp_path: Path) -> None:
    ext = _ext(_vendors())
    with pytest.raises(KeyError):
        ext.extract("http://nobody.example/x", tmp_path / "x")
    assert not (tmp_path / "x").exists()
    with pytest.raises(FileNotFoundError):
        ext.extract(A, tmp_path / "missing" / "a.xml")


def test_an_id_without_a_file_cannot_be_extracted(tmp_path: Path) -> None:
    x3p = _read_with([URI], {})
    with pytest.raises(X3pFormatError, match="without a file"):
        x3p.extensions.extract(URI, tmp_path / "x")
    assert not (tmp_path / "x").exists()


# --- unpacking when asked for -----------------------------------------------------------------

PAYLOAD = b"vendor payload " * 200
BIG = "http://www.vendor.com/mypath/big.dat"
OTHER = "http://www.vendor.com/mypath/other.dat"


def _blob_with_payload() -> bytes:
    x3p = _file()
    x3p.extensions[BIG] = PAYLOAD
    x3p.extensions[OTHER] = b"ok"
    return x3p.dumps()


def _damage(blob: bytes, name: str) -> bytes:
    """Return the file with a byte in the middle of the compressed member ``name`` changed."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        info = next(i for i in archive.infolist() if i.filename.endswith(name))
        start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    damaged = bytearray(blob)
    damaged[start + info.compress_size // 2] ^= 0xFF
    return bytes(damaged)


def _packed(x3p: X3pFile) -> list[bool]:
    return [isinstance(content, PackedMember) for content in _ext(x3p)._store.files.values()]


def test_a_file_that_was_read_keeps_its_extensions_packed() -> None:
    x3p = X3pFile.loads(_blob_with_payload())
    assert _packed(x3p) == [True, True]
    assert x3p.extensions[BIG] == PAYLOAD
    assert x3p.extensions[BIG] == PAYLOAD  # again, unpacked anew
    assert X3pFile.loads(x3p.dumps()) == x3p


def test_names_need_no_unpacking_of_a_damaged_member() -> None:
    x3p = X3pFile.loads(_damage(_blob_with_payload(), "big.dat"))
    assert list(x3p.extensions) == [BIG, OTHER]
    assert BIG in x3p.extensions
    assert len(_ext(x3p).below("www.vendor.com")) == 2
    assert x3p.extensions[OTHER] == b"ok"
    del x3p.extensions[BIG]
    assert list(x3p.extensions) == [OTHER]


def test_a_damaged_member_is_reported_when_it_is_asked_for(tmp_path: Path) -> None:
    x3p = X3pFile.loads(_damage(_blob_with_payload(), "big.dat"))
    with pytest.raises(X3pFormatError, match=r"big\.dat"):
        x3p.extensions[BIG]
    with pytest.raises(X3pFormatError, match=r"big\.dat"):
        x3p.extensions.extract(BIG, tmp_path / "big.dat")
    with pytest.raises(X3pFormatError, match=r"big\.dat"):
        dict(_ext(x3p).below("www.vendor.com"))
    with pytest.raises(X3pFormatError, match=r"big\.dat"):
        x3p.dumps()


def test_clearing_does_not_unpack_a_damaged_member() -> None:
    x3p = X3pFile.loads(_damage(_blob_with_payload(), "big.dat"))
    x3p.extensions.clear()
    assert len(x3p.extensions) == 0
    X3pFile.loads(x3p.dumps())


def test_a_member_with_another_compression_is_read_at_once() -> None:
    members = unzip(_with_id(build(), URI))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
        info = zipfile.ZipInfo("member")
        info.filename = SPEC_NAME  # set later, as a backslash would be converted on Windows
        info.compress_type = zipfile.ZIP_BZIP2
        archive.writestr(info, PAYLOAD)
    x3p = X3pFile.loads(buffer.getvalue())
    assert _packed(x3p) == [False]
    assert x3p.extensions[URI] == PAYLOAD


def test_a_stored_member_is_packed_too() -> None:
    members = unzip(_with_id(build(), URI))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
        info = zipfile.ZipInfo("member")
        info.filename = SPEC_NAME
        info.compress_type = zipfile.ZIP_STORED
        archive.writestr(info, b"as is")
    x3p = X3pFile.loads(buffer.getvalue())
    assert _packed(x3p) == [True]
    assert x3p.extensions[URI] == b"as is"


# --- where the content comes from --------------------------------------------------------------


def test_the_content_may_be_bytes_like() -> None:
    x3p = _file()
    x3p.extensions.add("http://a.b", "a.bin", bytearray(b"12"))
    x3p.extensions.add("http://a.b", "b.bin", memoryview(b"34"))
    assert dict(x3p.extensions) == {"http://a.b/a.bin": b"12", "http://a.b/b.bin": b"34"}
    assert all(type(content) is bytes for content in x3p.extensions.values())


def test_the_content_may_be_a_file_given_by_a_path(tmp_path: Path) -> None:
    source = tmp_path / "extension.dat"
    source.write_bytes(b"\x00file\xff")
    x3p = _file()
    x3p.extensions.add("http://a.b", "from_path.dat", source)
    x3p.extensions.add("http://a.b", "from_str.dat", str(source))
    assert dict(x3p.extensions) == {
        "http://a.b/from_path.dat": b"\x00file\xff",
        "http://a.b/from_str.dat": b"\x00file\xff",
    }
    # The content was read at once: changing the file afterwards changes nothing.
    source.write_bytes(b"changed")
    assert X3pFile.loads(x3p.dumps()).extensions["http://a.b/from_path.dat"] == b"\x00file\xff"


def test_the_content_may_be_a_stream(tmp_path: Path) -> None:
    stream = io.BytesIO(b"skipped-stream data")
    stream.seek(len(b"skipped-"))
    x3p = _file()
    x3p.extensions.add("http://a.b", "stream.dat", stream)
    assert x3p.extensions["http://a.b/stream.dat"] == b"stream data"
    assert not stream.closed
    source = tmp_path / "x.dat"
    source.write_bytes(b"open file")
    with source.open("rb") as handle:
        x3p.extensions.add("http://a.b", "handle.dat", handle)
        assert not handle.closed
    assert x3p.extensions["http://a.b/handle.dat"] == b"open file"


def test_a_string_is_a_path_and_not_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _file().extensions.add("http://www.vendor.com", "a.xml", str(tmp_path / "missing.dat"))


def test_a_missing_file_leaves_the_file_unchanged(tmp_path: Path) -> None:
    x3p = _file()
    with pytest.raises(OSError, match="missing"):
        x3p.extensions.add("http://www.vendor.com", "a.xml", tmp_path / "missing.dat")
    assert len(x3p.extensions) == 0


def test_an_invalid_path_is_rejected_before_the_source_is_read() -> None:
    class Unreadable(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise AssertionError("must not be read")

    with pytest.raises(X3pFormatError):
        _file().extensions.add("http://a.b", "", Unreadable())


def test_extensions_given_to_the_constructor_are_written() -> None:
    x3p = _file()
    extensions = VendorExtensions()
    extensions["http://a.b/c.bin"] = b"1"
    given = Surface.from_stacked(header=x3p.header, z=stacked(x3p), extensions=extensions)
    assert X3pFile.loads(given.dumps()).extensions == {"http://a.b/c.bin": b"1"}
