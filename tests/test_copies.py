# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""A copy is independent of the file it was made from, except for its arrays."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from helpers import absolute, array_file, stacked
from x3pio import (
    CoordinateSystem,
    DataType,
    Header,
    Metadata,
    Placement,
    PointCloud,
    Surface,
    SurfaceLayer,
    X3pFile,
)

ROTATION = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _file(*, rotation: bool = False, absolute: bool = False) -> X3pFile:
    if absolute:
        return PointCloud.from_points(
            np.random.default_rng(1).uniform(0, 1e-3, (5, 3)),
            placement=Placement(ROTATION),
            coordinate_system=CoordinateSystem.VIEW,
        )
    x3p = Surface.from_array(
        np.arange(6.0).reshape(2, 3) * 1e-6,
        x_scale=1e-6,
        y_scale=1e-6,
        metadata=Metadata(comment="original"),
        placement=Placement(ROTATION if rotation else None, offset=[0.01, 0.0, 0.002]),
        coordinate_system=CoordinateSystem.VIEW,
    )
    x3p.extensions["http://a.b/c.bin"] = b"1"
    return x3p


COPIES: dict[str, Callable[[X3pFile], X3pFile]] = {
    "with_metadata": lambda x3p: x3p.with_metadata(comment="changed"),
    "with_z_type": lambda x3p: x3p.with_z_type(DataType.INT32),
    "globalized": lambda x3p: x3p.globalized(),
}


@pytest.mark.parametrize("rotation", [False, True])
@pytest.mark.parametrize("name", list(COPIES))
def test_the_metadata_and_the_extensions_are_independent(name: str, rotation: bool) -> None:
    original = _file(rotation=rotation)
    snapshot = X3pFile.loads(original.dumps())
    copy = COPIES[name](original)

    assert copy.metadata is not original.metadata
    assert copy.extensions is not original.extensions
    with pytest.raises(FrozenInstanceError):
        copy.header.z.increment = 3.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        copy.placement = Placement.translation(5.0, 5.0, 5.0)  # type: ignore[misc]
    assert copy.metadata is not None
    copy.metadata.comment = "edited"
    copy.metadata.instrument.model = "edited"
    copy.extensions["http://a.b/new.bin"] = b"2"
    del copy.extensions["http://a.b/c.bin"]

    assert original == X3pFile.loads(original.dumps()) == snapshot


@pytest.mark.parametrize("name", ["with_metadata", "with_z_type"])
def test_the_arrays_are_shared_as_read_only_views(name: str) -> None:
    original = _file()
    copy = COPIES[name](original)
    assert stacked(copy) is not stacked(original)
    assert np.shares_memory(stacked(copy), stacked(original))
    assert not stacked(copy).flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        stacked(copy)[0, 0, 0] = 1.0
    assert stacked(original).flags.writeable
    stacked(original)[0, 0, 0] = 7.0  # the original stays writeable, and the copy shows it
    assert stacked(copy)[0, 0, 0] == 7.0


def test_the_absolute_coordinates_are_shared_as_read_only_views_too() -> None:
    original = _file(absolute=True)
    copy = original.with_metadata(comment="x")
    for array, source in (
        (absolute(copy, "x"), absolute(original, "x")),
        (absolute(copy, "y"), absolute(original, "y")),
        (stacked(copy), stacked(original)),
    ):
        assert np.shares_memory(array, source)
        assert not array.flags.writeable
        assert source.flags.writeable


@pytest.mark.parametrize("rotation", [False, True])
def test_the_global_copy_has_arrays_of_its_own(rotation: bool) -> None:
    original = _file(rotation=rotation)
    copy = original.globalized()
    assert not np.shares_memory(stacked(copy), stacked(original))
    assert stacked(copy).flags.writeable


def test_a_copy_can_be_changed_after_its_arrays_are_copied() -> None:
    original = _file()
    layers = original.with_metadata(comment="x").layers
    assert all(isinstance(layer, SurfaceLayer) for layer in layers)
    copy = Surface.from_layers(layers)  # type: ignore[arg-type]
    stacked(copy)[0, 0, 0] = 9.0
    assert stacked(original)[0, 0, 0] != 9.0
    assert stacked(X3pFile.loads(copy.dumps()))[0, 0, 0] == 9.0


@pytest.mark.parametrize("name", list(COPIES))
def test_a_copy_can_be_written_and_read_back(name: str) -> None:
    copy = COPIES[name](_file())
    again = X3pFile.loads(copy.dumps())
    assert again.extensions == copy.extensions


def test_a_copy_of_a_file_without_metadata_or_extensions() -> None:
    original = _file()
    original.metadata = None
    original.extensions.clear()
    copy = original.with_z_type(DataType.FLOAT32)
    assert copy.metadata is None
    assert len(copy.extensions) == 0
    assert isinstance(copy.header, Header)


def test_the_metadata_given_to_a_new_file_is_copied() -> None:
    metadata = Metadata(comment="mine")
    x3p = array_file(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0, metadata=metadata)
    cloud = PointCloud.from_points(np.zeros((2, 3)), metadata=metadata)
    metadata.comment = "changed later"
    assert x3p.metadata is not None
    assert x3p.metadata.comment == "mine"
    assert cloud.metadata is not None
    assert cloud.metadata.comment == "mine"
