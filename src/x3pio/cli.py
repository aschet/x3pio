# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Command line interface.

``x3pio info FILE`` prints the header, the metadata and the sizes of a file, without reading its
data.

``x3pio convert SOURCE DESTINATION`` changes the storage form (``-s``, ``xml`` or ``binary``),
the z data type (``-t``) and the revision (``-r``) independently. Whatever is not given is kept
from the source file, and deviations of the source are repaired. ``-f``/``--fix`` drops what
cannot be written instead of failing: metadata without a date or a probing type, and the vendor
extensions when converting to ``ISO5436_2000``.

The command line is also runnable as ``python -m x3pio``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime

from . import __version__
from .application import info, read
from .exceptions import X3pError
from .model.datatypes import DataType
from .model.header import Axis, FeatureType, Revision
from .model.info import FileDescription, FileInfo
from .model.storage import DataStorage

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="x3pio",
        description="Inspect and convert ISO 25178-72 x3p surface, profile and point cloud files.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="print x3p file information")
    info_parser.add_argument("path")

    convert_parser = subparsers.add_parser(
        "convert", help="convert the storage form, the z data type and/or the revision"
    )
    convert_parser.add_argument("source")
    convert_parser.add_argument("destination")
    convert_parser.add_argument(
        "-s",
        "--storage",
        choices=[storage.name.lower() for storage in DataStorage],
        help="store the coordinates as XML text or in a binary file",
    )
    convert_parser.add_argument(
        "-t",
        "--type",
        dest="data_type",
        choices=[data_type.name.lower() for data_type in DataType],
        help="convert the storage type of the z coordinates",
    )
    convert_parser.add_argument(
        "-r",
        "--revision",
        choices=[revision.name for revision in Revision],
        help="convert to the revision of the standard",
    )
    convert_parser.add_argument(
        "-f",
        "--fix",
        action="store_true",
        help="drop what cannot be written instead of failing: metadata without a date or a "
        "probing type, and the vendor extensions when converting to ISO5436_2000",
    )

    return parser


def _describe_axis(axis: Axis, offset: float) -> str:
    kind = "incremental" if axis.is_incremental else f"absolute, {axis.data_type.name.lower()}"
    return f"{kind}, Increment={axis.increment:g}, Offset={offset:g}"


def _date(value: datetime | None) -> str:
    return value.isoformat() if value is not None else "(not recorded)"


def _head_fields(description: FileDescription) -> list[tuple[str, str]]:
    return [
        ("Revision", description.revision.value),
        ("FeatureType", description.feature_type.value),
        ("Storage", description.storage.name.lower()),
    ]


def _size_fields(summary: FileInfo) -> list[tuple[str, str]]:
    if summary.feature_type is FeatureType.PROFILE:
        return [
            ("SizeX", str(summary.shape[0])),
            ("SizeY", "1"),
            ("SizeZ", str(summary.layer_count)),
        ]
    if summary.feature_type is FeatureType.SURFACE:
        return [
            ("SizeX", str(summary.shape[1])),
            ("SizeY", str(summary.shape[0])),
            ("SizeZ", str(summary.layer_count)),
        ]
    return [("ListDimension", str(summary.shape[0]))]


def _tail_fields(description: FileDescription) -> list[tuple[str, str]]:
    header = description.header
    fields = [
        ("CX", _describe_axis(header.x, description.placement.offset[0])),
        ("CY", _describe_axis(header.y, description.placement.offset[1])),
        ("CZ", _describe_axis(header.z, description.placement.offset[2])),
    ]
    if description.placement.has_rotation:
        fields.append(
            ("Rotation", " ".join(f"{value:g}" for value in description.placement.rotation.flat))
        )
    metadata = description.metadata
    if metadata is None:
        fields.append(("Record2", "(none)"))
    else:
        fields += [
            ("Date", _date(metadata.date)),
            ("Creator", metadata.creator if metadata.creator is not None else ""),
            ("Manufacturer", metadata.instrument.manufacturer),
            ("Model", metadata.instrument.model),
            ("Serial", metadata.instrument.serial),
            ("Version", metadata.instrument.version),
            ("CalibrationDate", _date(metadata.calibration_date)),
            (
                "ProbingType",
                metadata.probing_system.type.value if metadata.probing_system.type else "",
            ),
            ("Identification", metadata.probing_system.identification),
            ("Comment", metadata.comment if metadata.comment is not None else ""),
        ]
    extensions = description.extensions
    if description.revision.has_vendor_ids:
        fields += [
            ("VendorSpecificID", uri if extensions._has_file(uri) else f"{uri} (no file)")
            for uri in extensions
        ]
    else:
        ids, files = extensions._parts()
        fields += [("VendorSpecificID", vendor) for vendor in ids]
        fields += [("VendorFile", path) for path in files]
    return fields


def _info_fields(summary: FileInfo) -> list[tuple[str, str]]:
    return [*_head_fields(summary), *_size_fields(summary), *_tail_fields(summary)]


def _print_info(path: str) -> None:
    fields = _info_fields(info(path))
    width = max(len(name) for name, _ in fields)
    for name, value in fields:
        print(f"{name:<{width}} = {value}")


def _convert(args: argparse.Namespace) -> None:
    x3p = read(args.source)
    if args.revision is not None:
        x3p = x3p.with_revision(
            Revision[args.revision],
            drop_extensions=args.fix,
        )
    if args.data_type is not None:
        x3p = x3p.with_z_type(DataType[args.data_type.upper()])
    metadata = x3p.metadata
    if (
        args.fix
        and metadata is not None
        and (metadata.date is None or metadata.probing_system.type is None)
    ):
        x3p.metadata = None
    storage = DataStorage[args.storage.upper()] if args.storage is not None else None
    x3p.save(args.destination, storage=storage)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    :param argv: Arguments to parse; defaults to :data:`sys.argv`.
    :returns: The process exit status.
    """
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "info":
            _print_info(args.path)
            return 0
        if args.command == "convert":
            _convert(args)
            return 0
    # Deliberately narrow: only known, expected failure modes (malformed x3p
    # content, filesystem errors) get the friendly one-line message. Anything
    # else is a real bug and should surface as an actual traceback.
    except (X3pError, OSError) as error:
        print(f"x3pio: error: {error}", file=sys.stderr)
        return 1
    return 1
