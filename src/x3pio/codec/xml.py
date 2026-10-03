# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Parser and builder of ``main.xml``.

Parsing is deliberately tolerant: it accepts the deviations from the schema
that exist in files of other software and corrects them without a word. Building
always emits a document that is valid against the schema of the standard.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

# Only used to build documents; parsing goes through defusedxml.
from xml.etree import ElementTree as ET  # nosec B405

import numpy as np
from defusedxml import ElementTree as DefusedET
from defusedxml.common import DefusedXmlException

from ..exceptions import X3pFormatError
from ..model.datatypes import DataType
from ..model.geometry import Placement
from ..model.header import (
    Axis,
    AxisType,
    FeatureType,
    Header,
    Instrument,
    Metadata,
    ProbingSystem,
    ProbingType,
    Revision,
)
from ..model.naming import CHECKSUM_FILE

__all__ = ["NAMESPACE", "DataLink", "Document", "build_main", "parse_datums", "parse_main"]

NAMESPACE = "http://www.opengps.eu/2008/ISO5436_2"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_SCHEMA_LOCATION = f"{NAMESPACE} {NAMESPACE}/ISO5436_2.xsd"

#: Local name the root element is accepted under.
_ROOT_NAMES = ("ISO5436_2",)

#: Characters that XML 1.0 forbids in text.
_XML_FORBIDDEN_RE = re.compile(r"[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")

#: The children of each element, in the order the schema defines them.
_RECORDS = ("Record1", "Record2", "Record3", "Record4", "VendorSpecificID")
_RECORD1 = ("Revision", "FeatureType", "Axes")
_AXES = ("CX", "CY", "CZ", "Rotation")
_AXIS = ("AxisType", "DataType", "Increment", "Offset")
_ROTATION = tuple(f"r{row}{col}" for row in (1, 2, 3) for col in (1, 2, 3))
_RECORD2 = ("Date", "Creator", "Instrument", "CalibrationDate", "ProbingSystem", "Comment")
_INSTRUMENT = ("Manufacturer", "Model", "Serial", "Version")
_PROBING = ("Type", "Identification")
_RECORD3 = ("MatrixDimension", "ListDimension", "DataLink", "DataList")
_MATRIX = ("SizeX", "SizeY", "SizeZ")
_DATALINK = ("PointDataLink", "MD5ChecksumPointData", "ValidPointsLink", "MD5ChecksumValidPoints")

#: How much a rotation element may exceed 1 and still be taken as rounding noise.
_ROTATION_SLACK = 1e-6


@dataclass
class DataLink:
    """The ``DataLink`` element: where the binary coordinates and validity bits are.

    The checksums are lower-case hexadecimal digests, or ``None`` if the file
    gave none.
    """

    point_data: str
    point_data_md5: str | None
    valid_points: str | None
    valid_points_md5: str | None


@dataclass
class Document:
    """The content of a ``main.xml``, with deviations from the schema already corrected.

    :param header: The axes of ``Record1``.
    :param revision: The ``Revision`` of ``Record1``.
    :param placement: The offsets of the axes and the rotation of ``Record1``.
    :param feature_type: The class of the data, corrected to agree with the data.
    :param metadata: ``Record2``, or ``None`` if the file has none.
    :param size: ``(SizeX, SizeY, SizeZ)`` of a matrix, or ``None`` for a list.
    :param count: ``ListDimension``, or ``None`` for a matrix.
    :param datums: The text of every ``Datum`` of a ``DataList``, or ``None``
        if the coordinates are in a binary file.
    :param link: The ``DataLink`` to the binary files, or ``None`` for a
        ``DataList``.
    :param checksum_file: Path of the checksum file of ``main.xml``, which belongs to the format.
    :param vendor_ids: Every ``VendorSpecificID``.
    """

    header: Header
    revision: Revision
    placement: Placement
    feature_type: FeatureType
    metadata: Metadata | None
    size: tuple[int, int, int] | None
    count: int | None
    datums: list[str] | None
    link: DataLink | None
    checksum_file: str
    vendor_ids: list[str]

    @property
    def stored_shape(self) -> tuple[int, ...]:
        """The shape of the stored arrays: ``(layers, rows, columns)``, or ``(points,)`` of a list.

        :raises X3pFormatError: If ``Record3`` holds neither a matrix nor a list.
        """
        if self.size is not None:
            size_u, size_v, size_w = self.size
            return (size_w, size_v, size_u)
        if self.count is not None:
            return (self.count,)
        raise X3pFormatError("Record3 holds neither a MatrixDimension nor a ListDimension")

    @property
    def layer_count(self) -> int:
        """The number of layers: ``SizeZ`` of a matrix, and one for a list."""
        return self.stored_shape[0] if self.size is not None else 1

    @property
    def layer_shape(self) -> tuple[int, ...]:
        """The shape of a layer: ``(rows, columns)``, and ``(points,)`` for a profile or a list."""
        shape = self.stored_shape
        if self.size is None:
            return shape
        return shape[2:] if self.feature_type is FeatureType.PROFILE else shape[1:]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _local(tag: object) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _text(element: ET.Element) -> str:
    return (element.text or "").strip()


def _children(element: ET.Element, known: Sequence[str]) -> dict[str, ET.Element]:
    """Index the known children by name, in any order; unknown and repeated ones are ignored."""
    found: dict[str, ET.Element] = {}
    for child in element:
        name = _local(child.tag)
        if name in known and name not in found:
            found[name] = child
    return found


def _require(children: dict[str, ET.Element], name: str, where: str) -> ET.Element:
    if name not in children:
        raise X3pFormatError(f"Missing element <{name}> in <{where}>")
    return children[name]


def _number(text: str, what: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise X3pFormatError(f"{what} is not a number: {text!r}") from None
    if not math.isfinite(value):
        raise X3pFormatError(f"{what} is not a finite number: {text!r}")
    return value


def _count(text: str, what: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise X3pFormatError(f"{what} is not a whole number: {text!r}") from None
    if value < 0:
        raise X3pFormatError(f"{what} must not be negative: {text!r}")
    return value


def _parse_axis(element: ET.Element, name: str) -> tuple[Axis, float]:
    children = _children(element, _AXIS)
    type_text = _text(_require(children, "AxisType", name))
    try:
        axis_type = AxisType(type_text)
    except ValueError:
        raise X3pFormatError(f"{name}: unknown AxisType {type_text!r}") from None

    data_text = _text(children["DataType"]) if "DataType" in children else ""
    if data_text:
        try:
            data_type = DataType(data_text)
        except ValueError:
            raise X3pFormatError(f"{name}: unknown DataType {data_text!r}") from None
    elif axis_type is AxisType.ABSOLUTE:
        raise X3pFormatError(f"{name}: an absolute axis needs a DataType")
    else:
        data_type = DataType.FLOAT64  # an incremental axis does not use it

    increment_text = _text(children["Increment"]) if "Increment" in children else ""
    # Older files have no increment for the z axis; it is 1.
    increment = _number(increment_text, f"{name}: Increment") if increment_text else 1.0
    if increment == 0:
        raise X3pFormatError(f"{name}: Increment must not be zero")

    offset = 0.0
    if "Offset" in children:
        offset_text = _text(children["Offset"])
        if offset_text:
            offset = _number(offset_text, f"{name}: Offset")

    return Axis(axis_type, data_type, increment), offset


def _parse_rotation(element: ET.Element) -> np.ndarray:
    children = _children(element, _ROTATION)
    values = np.empty(9)
    for index, name in enumerate(_ROTATION):
        values[index] = _number(_text(_require(children, name, "Rotation")), f"Rotation {name}")
    outside = np.abs(values) > 1.0
    if np.any(np.abs(values[outside]) > 1.0 + _ROTATION_SLACK):
        raise X3pFormatError("Rotation elements must lie within [-1, 1]")
    return np.clip(values, -1.0, 1.0).reshape(3, 3)


def _parse_datetime(element: ET.Element | None) -> datetime | None:
    if element is None:
        return None
    try:
        value = datetime.fromisoformat(_text(element))
    except ValueError:
        return None
    if value.tzinfo is None:
        # ISO 8601: a time without any relation to UTC is a local time.
        value = _local_time(value) or value
    return value


def _offset_is_valid(value: datetime) -> bool:
    """Return whether the time zone offset of ``value`` is whole minutes within +-14 hours."""
    offset = value.utcoffset()
    return (
        offset is not None
        and offset.total_seconds() % 60 == 0
        and abs(offset) <= timedelta(hours=14)
    )


def _local_time(value: datetime) -> datetime | None:
    """Return a time without a zone as a local time, or ``None`` if it cannot be."""
    try:
        local = value.astimezone()
    except (ValueError, OverflowError, OSError):
        return None
    return local if _offset_is_valid(local) else None


def _format_datetime(value: datetime, what: str) -> str:
    """Format a date and time as the standard gives it: with a decimal fraction and a time zone.

    The fraction has as many digits as it needs, at least one, such as ``09.6`` or ``09.0``. A
    value without a time zone is taken as a local time, as ISO 8601 says, and written with the
    offset of the local time zone at that time.
    """
    if value.utcoffset() is None:
        local = _local_time(value)  # ISO 8601: a time without a zone is a local time
        if local is None:
            raise X3pFormatError(
                f"The {what} has no time zone, and cannot be taken as a local time"
            )
        value = local
    elif not _offset_is_valid(value):
        raise X3pFormatError(
            f"The {what} has the time zone offset {value.utcoffset()}, which has to be whole "
            "minutes within +-14:00"
        )
    text = value.isoformat(timespec="microseconds")
    stamp, zone = text[:-6], text[-6:]
    stamp = stamp.rstrip("0")
    return stamp + ("0" if stamp.endswith(".") else "") + zone


def _parse_metadata(element: ET.Element) -> Metadata:
    children = _children(element, _RECORD2)
    date = _parse_datetime(children.get("Date"))
    calibration = _parse_datetime(children.get("CalibrationDate"))
    creator = _text(children["Creator"]) if "Creator" in children else None
    comment = (children["Comment"].text or "") if "Comment" in children else None

    instrument = Instrument()
    if "Instrument" in children:
        parts = _children(children["Instrument"], _INSTRUMENT)
        instrument = Instrument(
            manufacturer=_text(parts["Manufacturer"]) if "Manufacturer" in parts else "",
            model=_text(parts["Model"]) if "Model" in parts else "",
            serial=_text(parts["Serial"]) if "Serial" in parts else "",
            version=_text(parts["Version"]) if "Version" in parts else "",
        )

    probing = ProbingSystem(type=None)
    if "ProbingSystem" in children:
        parts = _children(children["ProbingSystem"], _PROBING)
        probing_type = None
        if "Type" in parts:
            by_name = {member.value.lower(): member for member in ProbingType}
            probing_type = by_name.get(_text(parts["Type"]).lower())
        identification = _text(parts["Identification"]) if "Identification" in parts else ""
        probing = ProbingSystem(type=probing_type, identification=identification)

    return Metadata(date, creator, instrument, calibration, probing, comment)


def _parse_data_link(element: ET.Element) -> DataLink:
    children = _children(element, _DATALINK)
    point_data = _text(_require(children, "PointDataLink", "DataLink"))

    def digest(name: str) -> str | None:
        text = _text(children[name]).lower() if name in children else ""
        return text if re.fullmatch(r"[0-9a-f]{32}", text) else None

    valid_points = _text(children["ValidPointsLink"]) if "ValidPointsLink" in children else None
    return DataLink(
        point_data,
        digest("MD5ChecksumPointData"),
        valid_points or None,
        digest("MD5ChecksumValidPoints"),
    )


def parse_main(data: bytes) -> Document:
    """Parse a ``main.xml``, tolerating the deviations from the schema that other software makes.

    :param data: The content of the document.
    :raises X3pFormatError: If the document is not XML, or is damaged beyond
        what can be corrected.
    """
    try:
        root = DefusedET.fromstring(data)
    except (ET.ParseError, DefusedXmlException, ValueError) as error:
        raise X3pFormatError(f"main.xml is not valid XML: {error}") from error

    root_name = _local(root.tag)
    if root_name not in _ROOT_NAMES:
        raise X3pFormatError(f"main.xml has the unexpected root element <{root_name}>")

    records = _children(root, _RECORDS)
    record1 = _children(_require(records, "Record1", root_name), _RECORD1)

    feature_text = _text(_require(record1, "FeatureType", "Record1")).upper()
    try:
        feature_type = FeatureType(feature_text)
    except ValueError:
        raise X3pFormatError(f"Unknown FeatureType {feature_text!r}") from None

    axes = _children(_require(record1, "Axes", "Record1"), _AXES)
    (x, ox), (y, oy), (z, oz) = (
        _parse_axis(_require(axes, name, "Axes"), name) for name in _AXES[:3]
    )
    if z.axis_type is AxisType.INCREMENTAL:
        raise X3pFormatError("CZ: the z axis must be absolute")
    rotation = _parse_rotation(axes["Rotation"]) if "Rotation" in axes else np.eye(3)
    placement = Placement._unchecked(rotation, np.array([ox, oy, oz]))

    metadata = _parse_metadata(records["Record2"]) if "Record2" in records else None

    record3 = _children(_require(records, "Record3", root_name), _RECORD3)
    size: tuple[int, int, int] | None = None
    count: int | None = None
    if ("MatrixDimension" in record3) == ("ListDimension" in record3):
        raise X3pFormatError("Record3 needs exactly one of MatrixDimension and ListDimension")
    if "MatrixDimension" in record3:
        sizes = _children(record3["MatrixDimension"], _MATRIX)
        a, b, c = (
            _count(_text(_require(sizes, name, "MatrixDimension")), name) for name in _MATRIX
        )
        size = (a, b, c)
    else:
        count = _count(_text(record3["ListDimension"]), "ListDimension")

    datums: list[str] | None = None
    link: DataLink | None = None
    if ("DataLink" in record3) == ("DataList" in record3):
        raise X3pFormatError("Record3 needs exactly one of DataLink and DataList")
    if "DataLink" in record3:
        link = _parse_data_link(record3["DataLink"])
    else:
        datum_elements = _children_list(record3["DataList"], "Datum")
        datums = [_text(datum) for datum in datum_elements]

    feature_type = _reconcile_feature(feature_type, size, count)
    checksum_file = CHECKSUM_FILE
    if "Record4" in records:
        record4 = _children(records["Record4"], ("ChecksumFile",))
        checksum_file = _text(record4["ChecksumFile"]) if "ChecksumFile" in record4 else ""
        checksum_file = checksum_file or CHECKSUM_FILE
    vendor_ids = [text for e in root if _local(e.tag) == "VendorSpecificID" if (text := _text(e))]
    revision = Revision.resolve(_text(record1["Revision"]) if "Revision" in record1 else "")
    if len(vendor_ids) > 1:
        revision = Revision.ISO25178_72_2017_DAM1  # only Amd 1:2020 allows several IDs
    return Document(
        Header(x, y, z),
        revision,
        placement,
        feature_type,
        metadata,
        size,
        count,
        datums,
        link,
        checksum_file,
        vendor_ids,
    )


def _children_list(element: ET.Element, name: str) -> list[ET.Element]:
    """Return all children called ``name``."""
    return [child for child in element if _local(child.tag) == name]


def _reconcile_feature(
    feature_type: FeatureType, size: tuple[int, int, int] | None, count: int | None
) -> FeatureType:
    """Make the feature type agree with the shape of the data, which decides how it is read."""
    if count is not None:
        return FeatureType.POINT_CLOUD
    if feature_type is FeatureType.POINT_CLOUD:
        return FeatureType.SURFACE
    if feature_type is FeatureType.PROFILE and size is not None and size[1] != 1:
        return FeatureType.SURFACE
    return feature_type


def parse_datums(texts: Sequence[str], columns: int) -> np.ndarray:
    """Parse the text of the ``Datum`` elements of a ``DataList``.

    :param texts: The text of each ``Datum``; an empty one is a missing point.
    :param columns: Number of coordinates in a ``Datum``, which is the number
        of absolute axes.
    :returns: ``(len(texts), columns)`` array with ``NaN`` in a missing point.
    :raises X3pFormatError: If a ``Datum`` is not a list of ``columns`` numbers.
    """
    values = np.full((len(texts), columns), np.nan)
    for index, text in enumerate(texts):
        if not text:
            continue
        parts = text.split(";")
        if len(parts) != columns:
            raise X3pFormatError(
                f"Datum {index + 1} holds {len(parts)} value(s), but the axes need {columns}"
            )
        try:
            values[index] = [float(part) for part in parts]
        except ValueError:
            raise X3pFormatError(f"Datum {index + 1} is not a list of numbers: {text!r}") from None
    return values


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def _token(value: str, what: str) -> str:
    """Return ``value`` as the whitespace-collapsed text of an ``xsd:token`` element."""
    _check_text(value, what)
    return " ".join(value.split())


def _check_text(value: str, what: str) -> None:
    if _XML_FORBIDDEN_RE.search(value):
        raise X3pFormatError(f"{what} contains a character that XML cannot hold")


def _add(parent: ET.Element, name: str, text: str | None = None) -> ET.Element:
    element = ET.SubElement(parent, name)
    element.text = text
    return element


def _real(value: float) -> str:
    return repr(float(value))


def _add_axis(parent: ET.Element, name: str, axis: Axis, offset: float) -> None:
    element = _add(parent, name)
    _add(element, "AxisType", axis.axis_type.value)
    _add(element, "DataType", axis.data_type.value)
    _add(element, "Increment", _real(axis.increment))
    _add(element, "Offset", _real(offset))


def _add_metadata(parent: ET.Element, metadata: Metadata, revision: Revision) -> None:
    if metadata.date is None:
        raise X3pFormatError(
            "The metadata has no date, which the standard requires; "
            "set one with with_metadata(date=...)"
        )
    if metadata.probing_system.type is None:
        raise X3pFormatError(
            "The probing system has no type, which the standard requires; "
            "set one with with_metadata(probing_type=...)"
        )
    record2 = _add(parent, "Record2")
    _add(record2, "Date", _format_datetime(metadata.date, "date"))
    if metadata.creator is not None:
        _add(record2, "Creator", _token(metadata.creator, "The creator"))
    instrument = _add(record2, "Instrument")
    _add(instrument, "Manufacturer", _token(metadata.instrument.manufacturer, "The manufacturer"))
    _add(instrument, "Model", _token(metadata.instrument.model, "The model"))
    _add(instrument, "Serial", _token(metadata.instrument.serial, "The serial number"))
    _add(instrument, "Version", _token(metadata.instrument.version, "The version"))
    calibration_date = metadata.calibration_date
    if calibration_date is None and revision.requires_calibration_date:
        # Without one the file would not conform to the schema of its revision.
        calibration_date = metadata.date
    if calibration_date is not None:
        _add(record2, "CalibrationDate", _format_datetime(calibration_date, "calibration date"))
    probing = _add(record2, "ProbingSystem")
    _add(probing, "Type", metadata.probing_system.type.value)
    _add(
        probing,
        "Identification",
        _token(metadata.probing_system.identification, "The probing identification"),
    )
    if metadata.comment is not None:
        _check_text(metadata.comment, "The comment")
        _add(record2, "Comment", metadata.comment)


def build_main(
    *,
    header: Header,
    revision: Revision,
    placement: Placement,
    feature_type: FeatureType,
    metadata: Metadata | None,
    size: tuple[int, int, int] | None,
    count: int | None,
    datums: Sequence[str] | None,
    link: DataLink | None,
    vendor_ids: Sequence[str],
) -> bytes:
    """Build a ``main.xml`` that is valid against the schema of the standard.

    :param header: The axes of ``Record1``.
    :param revision: The revision, written as the ``Revision`` element.
    :param placement: The offsets of the axes and the rotation to write; the rotation is left out
        if it is the identity.
    :param feature_type: The class of the data.
    :param metadata: ``Record2``, or ``None`` to leave it out.
    :param size: ``(SizeX, SizeY, SizeZ)`` of a matrix; exactly one of
        ``size`` and ``count`` must be given.
    :param count: ``ListDimension`` of a list.
    :param datums: The text of every ``Datum`` for a ``DataList``, ``""`` for
        a missing point; exactly one of ``datums`` and ``link`` must be given.
    :param link: The ``DataLink`` to the binary files.
    :param vendor_ids: Every ``VendorSpecificID`` to write.
    :returns: The UTF-8 encoded document.
    :raises X3pFormatError: If a text cannot be held by XML or a required
        piece of metadata is missing.
    """
    ET.register_namespace("p", NAMESPACE)
    ET.register_namespace("xsi", _XSI)
    root = ET.Element(f"{{{NAMESPACE}}}ISO5436_2")
    root.set(f"{{{_XSI}}}schemaLocation", _SCHEMA_LOCATION)

    record1 = _add(root, "Record1")
    _add(record1, "Revision", revision.value)
    _add(record1, "FeatureType", feature_type.value)
    axes = _add(record1, "Axes")
    for name, axis, offset in (
        ("CX", header.x, placement.offset[0]),
        ("CY", header.y, placement.offset[1]),
        ("CZ", header.z, placement.offset[2]),
    ):
        _add_axis(axes, name, axis, float(offset))
    if placement.has_rotation:
        rotation = _add(axes, "Rotation")
        for index, name in enumerate(_ROTATION):
            _add(rotation, name, _real(placement.rotation.flat[index]))

    if metadata is not None:
        _add_metadata(root, metadata, revision)

    record3 = _add(root, "Record3")
    if size is not None:
        matrix = _add(record3, "MatrixDimension")
        for name, value in zip(_MATRIX, size, strict=True):
            _add(matrix, name, str(value))
    else:
        _add(record3, "ListDimension", str(count))
    if link is not None:
        data_link = _add(record3, "DataLink")
        _add(data_link, "PointDataLink", link.point_data)
        _add(data_link, "MD5ChecksumPointData", link.point_data_md5)
        if link.valid_points is not None:
            _add(data_link, "ValidPointsLink", link.valid_points)
            _add(data_link, "MD5ChecksumValidPoints", link.valid_points_md5)
    else:
        data_list = _add(record3, "DataList")
        for text in datums or ():
            _add(data_list, "Datum", text or None)

    record4 = _add(root, "Record4")
    _add(record4, "ChecksumFile", CHECKSUM_FILE)
    for uri in vendor_ids:
        _check_text(uri, "A VendorSpecificID")
        _add(root, "VendorSpecificID", uri.strip())

    if link is None:
        # A long DataList is kept compact: one Datum per line is all the layout it needs.
        ET.indent(root, space="  ")
    else:
        ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n").encode("utf-8")
