# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Helpers that build and tamper with x3p files for the tests."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal, overload

import numpy as np
import xmlschema

import x3pio

#: The schema of each revision: ISO 25178-72:2017 and Amd 1:2020.
SCHEMAS = {
    x3pio.Revision.ISO5436_2000: xmlschema.XMLSchema(
        Path(__file__).parent / "xsd" / "iso25178-72-2017.xsd"
    ),
    x3pio.Revision.ISO25178_72_2017_DAM1: xmlschema.XMLSchema(
        Path(__file__).parent / "xsd" / "iso25178-72-fdam1.xsd"
    ),
}

SURFACE = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]) * 1e-6


def build(
    storage: x3pio.DataStorage = x3pio.DataStorage.XML,
    data: np.ndarray | None = None,
    **kwargs: object,
) -> bytes:
    """Return a small surface written by x3pio, as XML text by default."""
    buffer = io.BytesIO()
    x3pio.write(
        buffer,
        SURFACE if data is None else data,
        x_scale=1e-6,
        y_scale=2e-6,
        storage=storage,
        **kwargs,  # type: ignore[arg-type]
    )
    return buffer.getvalue()


def unzip(blob: bytes) -> dict[str, bytes]:
    """Return the members of an x3p file by their path."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def rezip(members: Mapping[str, bytes]) -> bytes:
    """Build a zip archive from members, in the given order."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def edit_main(
    blob: bytes, transform: Callable[[str], str], *, refresh_checksum: bool = True
) -> bytes:
    """Return the file with its main.xml changed, and its checksum file kept consistent."""
    members = unzip(blob)
    text = members["main.xml"].decode("utf-8")
    changed = transform(text)
    assert changed != text, "the edit changed nothing"
    members["main.xml"] = changed.encode("utf-8")
    if refresh_checksum:
        digest = hashlib.md5(members["main.xml"], usedforsecurity=False).hexdigest()
        members["md5checksum.hex"] = f"{digest} *main.xml\n".encode("ascii")
    return rezip(members)


Edit = Callable[[str], str]


def replace(old: str, new: str, count: int = 1) -> Edit:
    """Return an edit that replaces text of main.xml, which must be present."""

    def edit(text: str) -> str:
        assert old in text, f"{old!r} not in main.xml"
        return text.replace(old, new, count)

    return edit


def drop(pattern: str) -> Edit:
    """Return an edit that removes the first match of a regular expression."""

    def edit(text: str) -> str:
        changed, count = re.subn(pattern, "", text, count=1, flags=re.DOTALL)
        assert count == 1, f"{pattern!r} not in main.xml"
        return changed

    return edit


def sub(pattern: str, replacement: str) -> Edit:
    """Return an edit that substitutes every match of a regular expression."""

    def edit(text: str) -> str:
        changed, count = re.subn(pattern, replacement, text)
        assert count, f"{pattern!r} not in main.xml"
        return changed

    return edit


def container_from_main(main: bytes) -> bytes:
    """Wrap a bare main.xml into an x3p container with its checksum file."""
    digest = hashlib.md5(main, usedforsecurity=False).hexdigest()
    return rezip({"main.xml": main, "md5checksum.hex": f"{digest} *main.xml\n".encode("ascii")})


def array_file(data: object, **kwargs: object) -> x3pio.X3pFile:
    """Return a profile or a surface of heights, by the number of dimensions of ``data``."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        return x3pio.Profile.from_array(array, **kwargs)  # type: ignore[arg-type]
    if array.ndim in (2, 3):
        return x3pio.Surface.from_array(array, **kwargs)  # type: ignore[arg-type]
    raise x3pio.X3pFormatError("Data must be a 1-D profile, a 2-D surface or a 3-D array of layers")


@overload
def stacked(file: x3pio.X3pFile, name: Literal["z"] = "z") -> np.ndarray: ...


@overload
def stacked(file: x3pio.X3pFile, name: Literal["x", "y"]) -> np.ndarray | None: ...


def stacked(file: x3pio.X3pFile, name: Literal["x", "y", "z"] = "z") -> np.ndarray | None:
    """Return the array of ``z``, ``x`` or ``y`` of all layers stacked (white box).

    The array is a view for a single layer, and a copy for several.
    """
    z, x, y = file._stacked()
    return {"z": z, "x": x, "y": y}[name]


def flat_points(
    file: x3pio.X3pFile,
    *,
    coordinate_system: x3pio.CoordinateSystem = x3pio.CoordinateSystem.GLOBAL,
    drop_invalid: bool = False,
) -> np.ndarray:
    """Return the points of all layers of a file as one ``(N, 3)`` array."""
    points = np.concatenate(
        [layer.points(coordinate_system=coordinate_system).reshape(-1, 3) for layer in file.layers]
    )
    return points[np.isfinite(points).all(axis=1)] if drop_invalid else points


def shaped_points(
    file: x3pio.X3pFile,
    *,
    coordinate_system: x3pio.CoordinateSystem = x3pio.CoordinateSystem.GLOBAL,
) -> np.ndarray:
    """Return the points of a file shaped like its layers stacked, with a last axis of three."""
    arrays = [layer.points(coordinate_system=coordinate_system) for layer in file.layers]
    return arrays[0] if isinstance(file, x3pio.PointCloud) else np.stack(arrays)


def absolute(file: x3pio.X3pFile, name: Literal["x", "y"]) -> np.ndarray:
    """Return the stored ``x`` or ``y`` of a file whose axis is absolute (white box)."""
    array = stacked(file, name)
    assert array is not None, f"The {name} axis has no coordinates"
    return array
