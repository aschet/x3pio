# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""``x3pio.info`` reads the description of a file without its data."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import numpy as np
import pytest

import x3pio
from helpers import build, rezip, unzip
from x3pio import (
    DataStorage,
    FeatureType,
    FileDescription,
    FileInfo,
    Metadata,
    Placement,
    PointCloud,
    Profile,
    Revision,
    Surface,
    X3pChecksumError,
    X3pFile,
    X3pFormatError,
)


def _blob(file: X3pFile) -> bytes:
    return file.dumps()


def _surface(**options: object) -> Surface:
    return Surface.from_array(
        np.arange(24.0).reshape(2, 3, 4),
        x_scale=1e-6,
        y_scale=2e-6,
        metadata=Metadata(comment="catalogue"),
        **options,  # type: ignore[arg-type]
    )


def test_the_summary_says_what_the_file_says() -> None:
    surface = _surface(
        placement=Placement.translation(1.0, 2.0, 3.0),
        coordinate_system=x3pio.CoordinateSystem.VIEW,
    )
    summary = x3pio.info(io.BytesIO(_blob(surface)))
    assert isinstance(summary, FileInfo)
    assert summary.feature_type is FeatureType.SURFACE
    assert summary.layer_count == 2
    assert summary.shape == (3, 4)  # rows and columns
    assert summary.header == surface.header
    assert summary.placement == surface.placement
    assert summary.metadata == surface.metadata
    assert summary.storage is DataStorage.BINARY
    assert summary.extensions == surface.extensions


def test_the_summary_is_that_of_the_file_that_is_read() -> None:
    for file in (
        _surface(),
        Profile.from_array(np.zeros((3, 5)), x_scale=1.0),
        PointCloud.from_points(np.zeros((7, 3))),
    ):
        blob = _blob(file)
        summary = x3pio.info(io.BytesIO(blob))
        read = X3pFile.loads(blob)
        assert summary.header == read.header
        assert summary.feature_type is read.feature_type
        assert summary.layer_count == len(read.layers)
        assert summary.shape == read.layers[0].z.shape
        assert summary.metadata == read.metadata


def test_the_storage_form_is_reported() -> None:
    assert x3pio.info(io.BytesIO(build(DataStorage.XML))).storage is DataStorage.XML
    assert x3pio.info(io.BytesIO(build(DataStorage.BINARY))).storage is DataStorage.BINARY


def test_the_summary_cannot_be_assigned() -> None:
    summary = x3pio.info(io.BytesIO(build()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.shape = (1,)  # type: ignore[misc]


@pytest.mark.parametrize("revision", list(Revision))
def test_the_extensions_are_listed_for_both_revisions(revision: Revision) -> None:
    surface = _surface(revision=revision)
    surface.extensions.add("http://a.b", "c.bin", b"1")
    summary = x3pio.info(io.BytesIO(_blob(surface)))
    assert summary.extensions == surface.extensions
    assert summary.revision is revision


def test_a_path_is_read_and_the_file_is_closed(tmp_path: Path) -> None:
    path = tmp_path / "a.x3p"
    _surface().save(path)
    assert x3pio.info(path).layer_count == 2
    assert x3pio.info(str(path)).shape == (3, 4)
    path.unlink()  # not held open (matters on Windows)


def test_a_stream_that_cannot_seek_is_read_whole() -> None:
    class Pipe(io.RawIOBase):
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)

        def readable(self) -> bool:
            return True

        def seekable(self) -> bool:
            return False

        def readinto(self, buffer: bytearray) -> int:  # type: ignore[override]
            data = self._buffer.read(len(buffer))
            buffer[: len(data)] = data
            return len(data)

    pipe = io.BufferedReader(Pipe(_blob(_surface())))  # type: ignore[type-var]
    summary = x3pio.info(pipe)
    assert summary.shape == (3, 4)


def test_the_data_is_not_read() -> None:
    members = unzip(_blob(_surface()))
    del members["bindata/data.bin"]
    blob = rezip(members)
    assert x3pio.info(io.BytesIO(blob)).shape == (3, 4)
    with pytest.raises(X3pFormatError, match="missing"):
        X3pFile.loads(blob)

    members["bindata/data.bin"] = b"damaged"
    assert x3pio.info(io.BytesIO(rezip(members))).layer_count == 2


def test_the_checksum_of_the_description_is_verified() -> None:
    members = unzip(_blob(_surface()))
    members["main.xml"] = members["main.xml"].replace(b"catalogue", b"catalogUe")
    blob = rezip(members)
    with pytest.raises(X3pChecksumError, match=r"main\.xml"):
        x3pio.info(io.BytesIO(blob))
    assert x3pio.info(io.BytesIO(blob), verify=False).metadata is not None


def test_something_that_is_not_an_x3p_file_is_refused() -> None:
    with pytest.raises(X3pFormatError, match="zip"):
        x3pio.info(io.BytesIO(b"not a zip"))
    with pytest.raises(X3pFormatError, match=r"main\.xml"):
        x3pio.info(io.BytesIO(rezip({"other.txt": b"x"})))


def test_a_file_without_metadata_has_none() -> None:
    surface = _surface()
    surface.metadata = None
    assert x3pio.info(io.BytesIO(_blob(surface))).metadata is None


SAMPLE_FILES = sorted(
    [
        *(Path(__file__).parent / "samples").rglob("*.x3p"),
        *(Path(__file__).parent / "interop_samples").rglob("*.x3p"),
    ]
)


@pytest.mark.parametrize("path", SAMPLE_FILES, ids=lambda path: path.name)
def test_info_agrees_with_read_on_the_sample_files(path: Path) -> None:
    summary = x3pio.info(path)
    file = x3pio.read(path)
    assert summary.feature_type is file.feature_type
    assert summary.header == file.header
    assert summary.revision is file.revision
    assert summary.placement == file.placement
    assert summary.storage is file.storage
    assert summary.metadata == file.metadata
    assert summary.extensions == file.extensions
    assert summary.layer_count == len(file.layers)
    assert summary.shape == file.layers[0].shape


@pytest.mark.parametrize("path", SAMPLE_FILES[:6], ids=lambda path: path.name)
def test_a_file_and_its_info_are_both_descriptions(path: Path) -> None:
    from x3pio import cli

    summary: FileDescription = x3pio.info(path)
    file: FileDescription = x3pio.read(path)
    assert cli._head_fields(file) == cli._head_fields(summary)
    assert cli._tail_fields(file) == cli._tail_fields(summary)
