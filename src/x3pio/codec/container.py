# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Reading and writing the zip container of an x3p file.

A container is a zip archive with a ``main.xml`` that describes the data, a small file with the
MD5 checksum of ``main.xml``, and, unless the numbers are written into ``main.xml`` as text,
a binary file with the coordinates and one with the validity of the points. The names of the
data files are links in ``main.xml``, and the recommended ones are ``bindata/data.bin`` and
``bindata/valid.bin``. Links must lead to files in the container and never out of it. The files
that a vendor adds are the vendor specific extensions. This module puts them together, in either
direction, and checks the checksums that the file states.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..exceptions import X3pChecksumError, X3pFormatError
from ..model.datatypes import DataTypeInfo, decode_raw, encode_raw, get_data_type
from ..model.extensions import VendorExtensions, content_of
from ..model.header import Axis, FeatureType, Revision
from ..model.info import FileInfo
from ..model.naming import (
    CHECKSUM_FILE,
    DATA_FILE,
    MAIN_XML,
    RESERVED_NAMES,
    VALID_FILE,
    local_member_path,
    vendor_extension_path,
)
from ..model.storage import DataStorage
from ..model.streams import PathOrStream
from ..model.validation import validate
from .archive import (
    Container,
    format_checksum_file,
    md5_hex,
    pack_validity,
    parse_checksum_file,
    unpack_validity,
    write_container,
)
from .xml import DataLink, Document, build_main, parse_datums, parse_main

if TYPE_CHECKING:
    from ..model.files import X3pFile

__all__ = ["describe", "parse", "records", "serialize"]


def parse(data: bytes, *, verify: bool) -> dict[str, Any]:
    """Parse the content of an x3p file into the fields of an :class:`~x3pio.X3pFile`.

    A set of known deviations from the standard is tolerated, see :mod:`x3pio.codec`.

    :raises X3pFormatError: If ``data`` is not an x3p file, or is damaged beyond what can
        be tolerated.
    :raises X3pChecksumError: If ``verify`` and a stored checksum does not match.
    """
    container = Container(data)
    main = container.read(MAIN_XML)
    document = parse_main(main)
    if verify:
        _verify_main(container, document, main)
    taken = _format_files(document)

    header = document.header
    absolute = [
        (name, axis)
        for name, axis in (("x", header.x), ("y", header.y), ("z", header.z))
        if not axis.is_incremental
    ]
    shape = document.stored_shape
    count = math.prod(shape)
    if document.feature_type is FeatureType.POINT_CLOUD and (
        header.x.is_incremental or header.y.is_incremental
    ):
        raise X3pFormatError("The axes of a point cloud must be absolute")

    valid: np.ndarray | None = None
    if document.link is not None:
        raw, valid = _read_binary(container, document.link, absolute, count, taken, verify)
    elif document.datums is not None:
        raw = _read_datums(document.datums, absolute, count)
    else:
        raise X3pFormatError("Record3 holds neither a DataLink nor a DataList")

    arrays = {name: decode_raw(raw[name], axis.increment).reshape(shape) for name, axis in absolute}
    z = arrays["z"]
    if valid is not None:
        # The validity file marks a whole point; what is stored for it is arbitrary.
        for coordinates in arrays.values():
            coordinates[~valid.reshape(shape)] = np.nan
    if document.feature_type is FeatureType.POINT_CLOUD:
        # A point cloud is a list that holds no unmeasured points; a file that has some is
        # tolerated by leaving them out.
        measured = np.isfinite(z)
        for coordinates in arrays.values():
            measured &= np.isfinite(coordinates)
        if not measured.all():
            arrays = {name: coordinates[measured] for name, coordinates in arrays.items()}
            z = arrays["z"]

    extensions = _read_extensions(container, document.vendor_ids, taken, document.revision)
    return {
        "header": header,
        "revision": document.revision,
        "placement": document.placement,
        "feature_type": document.feature_type,
        "z": z,
        "x": arrays.get("x"),
        "y": arrays.get("y"),
        "metadata": document.metadata,
        "extensions": extensions,
        "storage": DataStorage.XML if document.link is None else DataStorage.BINARY,
    }


def records(file: X3pFile) -> np.ndarray:
    """Return the coded coordinates in the layout of a binary data file, one record per point."""
    header = file.header
    data, x, y = file._cube()
    stored = {"x": x, "y": y, "z": data}
    absolute = [
        (name, axis)
        for name, axis in (("x", header.x), ("y", header.y), ("z", header.z))
        if not axis.is_incremental
    ]
    record = np.dtype([(name, get_data_type(axis.data_type).dtype) for name, axis in absolute])
    coded = np.empty(data.size, dtype=record)
    for name, axis in absolute:
        values = np.asarray(stored[name]).reshape(-1)
        coded[name] = encode_raw(values, get_data_type(axis.data_type), axis.increment)
    return coded


def serialize(file: X3pFile, storage: DataStorage | None) -> bytes:
    """Return the bytes of the file, in ``storage`` or in the form the file has."""
    validate(file)
    header = file.header
    use_binary = (file.storage if storage is None else storage) is DataStorage.BINARY

    # A point with a coordinate that is not finite (NaN or infinite) is not measured: it is
    # an empty Datum in XML, and marked in the validity file if an axis is an integer.
    data, x, y = file._cube()
    missing = ~np.isfinite(data)
    for stored in (x, y):
        if stored is not None:
            missing |= ~np.isfinite(stored)
    missing = missing.reshape(-1)
    coded = records(file)

    size: tuple[int, int, int] | None = None
    count: int | None = None
    if file.feature_type is FeatureType.POINT_CLOUD:
        count = data.shape[0]
    else:
        size = (data.shape[-1], data.shape[-2], data.shape[0])

    members: list[tuple[str, bytes]] = []
    datums: list[str] | None = None
    link: DataLink | None = None
    if use_binary:
        point_data = coded.tobytes()
        members.append((DATA_FILE, point_data))
        valid_path = valid_md5 = None
        # An integer has no value for a missing point: the validity file marks it.
        integer_axis = any(
            get_data_type(axis.data_type).is_integer
            for axis in (header.x, header.y, header.z)
            if not axis.is_incremental
        )
        if integer_axis and missing.any():
            valid_data = pack_validity(~missing)
            members.append((VALID_FILE, valid_data))
            valid_path, valid_md5 = VALID_FILE, md5_hex(valid_data)
        link = DataLink(DATA_FILE, md5_hex(point_data), valid_path, valid_md5)
    else:
        datums = _format_datums(coded, missing)

    vendor_ids, extension_files = file.extensions._parts()
    main = build_main(
        header=header,
        revision=file.revision,
        placement=file.placement,
        feature_type=file.feature_type,
        metadata=file.metadata,
        size=size,
        count=count,
        datums=datums,
        link=link,
        vendor_ids=vendor_ids,
    )
    checksum = format_checksum_file(md5_hex(main))
    extension_members = sorted(
        (name, content_of(stored)) for name, stored in extension_files.items()
    )
    return write_container(
        [
            (MAIN_XML, main),
            (CHECKSUM_FILE, checksum),
            *members,
            *extension_members,
        ]
    )


def describe(source: PathOrStream, *, verify: bool) -> FileInfo:
    """Read the description of an x3p file without its coordinates.

    The directory of the container, ``main.xml`` and the vendor specific files are read. The
    binary files of the coordinates are not, so their sizes and checksums are not checked.

    :raises X3pFormatError: If ``source`` is not an x3p file, or is damaged beyond what can
        be tolerated.
    :raises X3pChecksumError: If ``verify`` and the checksum of ``main.xml`` does not match.
    """
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as stream:
            return _describe(Container(stream), verify)
    return _describe(Container(source if source.seekable() else source.read()), verify)


def _describe(container: Container, verify: bool) -> FileInfo:
    main = container.read(MAIN_XML)
    document = parse_main(main)
    if verify:
        _verify_main(container, document, main)
    taken = _format_files(document)
    if document.link is not None:
        for link_name in (document.link.point_data, document.link.valid_points):
            if link_name is not None and (name := _member_or_none(link_name)) is not None:
                taken.add(name)
    return FileInfo(
        header=document.header,
        revision=document.revision,
        placement=document.placement,
        feature_type=document.feature_type,
        layer_count=document.layer_count,
        shape=document.layer_shape,
        storage=DataStorage.XML if document.link is None else DataStorage.BINARY,
        metadata=document.metadata,
        extensions=_read_extensions(container, document.vendor_ids, taken, document.revision),
    )


def _format_files(document: Document) -> set[str]:
    """Return the names of the files that belong to the format: what else is a vendor file."""
    taken = {*RESERVED_NAMES}
    if (name := _member_or_none(document.checksum_file)) is not None:
        taken.add(name)
    return taken


def _read_extensions(
    container: Container, ids: Sequence[str], taken: set[str], revision: Revision
) -> VendorExtensions:
    """Collect the vendor specific files of a file.

    Amd 1:2020 registers each file by its own ``VendorSpecificID``, which names it. In the
    standard before it a single ID stands for the extension as a whole, and its files lie in the
    container under any name except the ones of the format. A file that no ID registers is
    ignored, as the standard allows.
    """
    if not revision.has_vendor_ids:
        if not ids:
            return VendorExtensions(revision)
        files = {
            name: container.pack(name)
            for name in container.names
            if name not in taken and local_member_path(name) == name
        }
        return VendorExtensions._read_files(ids[0], files)
    extensions = VendorExtensions(revision)
    for uri in ids:
        stored = member = None
        try:
            member = vendor_extension_path(uri)
        except X3pFormatError:
            pass  # a URI without a path to a file, such as a bare domain, names no file
        else:
            name = member.replace("\\", "/")
            if container.has(name) and name not in RESERVED_NAMES:
                stored = container.pack(name)
                taken.add(name)
        extensions._announce(uri, stored, member)
    return extensions


def _member_or_none(link: str) -> str | None:
    try:
        return local_member_path(link)
    except X3pFormatError:
        return None


def _verify_main(container: Container, document: Document, main: bytes) -> None:
    """Check ``main.xml`` against the checksum file of ``Record4``, if there is one to check."""
    name = _member_or_none(document.checksum_file)
    if name is None or not container.has(name):
        return
    stored = parse_checksum_file(container.read(name))
    if stored is not None and stored != md5_hex(main):
        raise X3pChecksumError("The MD5 checksum of main.xml does not match its content")


def _read_binary(
    container: Container,
    link: DataLink,
    absolute: list[tuple[str, Axis]],
    count: int,
    taken: set[str],
    verify: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Read the coordinates and the validity mask from the binary files of a ``DataLink``."""
    record = np.dtype([(name, get_data_type(axis.data_type).dtype) for name, axis in absolute])

    path = local_member_path(link.point_data)
    if not container.has(path):
        raise X3pFormatError(f"The data file {path!r} is missing from the container")
    taken.add(path)
    expected = count * record.itemsize
    if container.size(path) != expected:
        raise X3pFormatError(
            f"The data file {path!r} holds {container.size(path)} bytes, "
            f"but the dimensions and axes need {expected}"
        )
    content = container.read(path)
    if verify and link.point_data_md5 is not None and md5_hex(content) != link.point_data_md5:
        raise X3pChecksumError(f"The MD5 checksum of {path!r} does not match its content")
    records = np.frombuffer(content, dtype=record, count=count)
    raw = {name: records[name] for name, _ in absolute}

    valid: np.ndarray | None = None
    if link.valid_points is not None:
        valid_path = local_member_path(link.valid_points)
        if not container.has(valid_path):
            raise X3pFormatError(f"The validity file {valid_path!r} is missing from the container")
        taken.add(valid_path)
        expected_bits = (count + 7) // 8
        if container.size(valid_path) != expected_bits:
            raise X3pFormatError(
                f"The validity file {valid_path!r} holds {container.size(valid_path)} bytes, "
                f"but the dimensions need {expected_bits}"
            )
        valid_content = container.read(valid_path)
        if (
            verify
            and link.valid_points_md5 is not None
            and md5_hex(valid_content) != link.valid_points_md5
        ):
            raise X3pChecksumError(f"The MD5 checksum of {valid_path!r} does not match its content")
        valid = unpack_validity(valid_content, count)
    return raw, valid


def _read_datums(
    datums: list[str], absolute: list[tuple[str, Axis]], count: int
) -> dict[str, np.ndarray]:
    """Read the coordinates from the ``Datum`` elements of a ``DataList``."""
    if len(datums) != count:
        raise X3pFormatError(f"The DataList holds {len(datums)} Datum elements, but {count} needed")
    values = parse_datums(datums, len(absolute))
    raw: dict[str, np.ndarray] = {}
    for column, (name, axis) in enumerate(absolute):
        data_type = get_data_type(axis.data_type)
        values_of_axis = values[:, column]
        if data_type.is_integer:
            _check_integers(values_of_axis, data_type, name)
        raw[name] = values_of_axis
    return raw


def _check_integers(values: np.ndarray, data_type: DataTypeInfo, name: str) -> None:
    finite = values[np.isfinite(values)]
    info = np.iinfo(data_type.dtype)
    if finite.size and (
        not np.array_equal(finite, np.rint(finite))
        or finite.min() < info.min
        or finite.max() > info.max
    ):
        raise X3pFormatError(
            f"The {name} values must be whole numbers within {info.min}..{info.max} "
            f"for the data type {data_type.type.name}"
        )


def _format_datums(records: np.ndarray, missing: np.ndarray) -> list[str]:
    """Format the coordinates as the text of ``Datum`` elements; a missing point is empty."""
    names = records.dtype.names or ()
    columns = [records[name].tolist() for name in names]
    texts: list[str] = []
    for index, values in enumerate(zip(*columns, strict=True)):
        if missing[index]:
            texts.append("")
            continue
        texts.append(
            ";".join(str(value) if isinstance(value, int) else repr(value) for value in values)
        )
    return texts
