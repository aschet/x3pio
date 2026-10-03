# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Zip archive, checksums and validity bit field of an x3p file.

The archive of a file that is read is opened without unpacking the members: an extension file
stays compressed until its content is asked for. Archivers of other operating systems add files
of their own to a zip, and those are not members. The validity file packs one bit per point.
Neither text of the standard says which bit of a byte comes first; x3pio counts from the least
significant bit, as the sample files of the openGPS library do.
"""

from __future__ import annotations

import hashlib
import io
import re
import struct
import zipfile
import zlib
from collections.abc import Iterable
from typing import IO

import numpy as np

from ..exceptions import X3pFormatError
from ..model.naming import MAIN_XML

__all__ = [
    "Container",
    "PackedMember",
    "format_checksum_file",
    "md5_hex",
    "pack_validity",
    "parse_checksum_file",
    "unpack_validity",
    "write_container",
]

#: Members that archivers and file managers of other operating systems add on their own, in lower
#: case since they are matched without regard to it: the folder of resource forks of macOS and
#: its ``._name`` companions of a file, ``.DS_Store``, and ``Thumbs.db`` and ``desktop.ini``
#: of Windows.
_JUNK_FOLDERS = frozenset({"__macosx"})
_JUNK_NAMES = frozenset({".ds_store", "thumbs.db", "desktop.ini"})
_JUNK_BASENAME_PREFIX = "._"


def _is_junk(name: str) -> bool:
    """Return whether a member path is a file that an operating system added on its own."""
    *folders, basename = name.lower().split("/")
    return (
        any(folder in _JUNK_FOLDERS for folder in folders)
        or basename in _JUNK_NAMES
        or basename.startswith(_JUNK_BASENAME_PREFIX)
    )


_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")

#: Zip timestamps are fixed so that identical content gives an identical file.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

#: The fixed part of a local file header of a zip archive; the last two fields are the
#: lengths of the name and the extra field that follow it.
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"
_ENCRYPTED = 0x1


def md5_hex(data: bytes) -> str:
    """Return the lower-case hexadecimal MD5 digest of ``data``."""
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def format_checksum_file(digest: str) -> bytes:
    """Format the content of the checksum file of ``main.xml``, like ``md5sum`` does."""
    return f"{digest} *{MAIN_XML}\n".encode("ascii")


def parse_checksum_file(data: bytes) -> str | None:
    """Extract the MD5 digest from the content of a checksum file.

    Accepts the ``md5sum`` layout (``<digest> *main.xml``) as well as a bare
    digest, in either case and with or without a trailing newline.

    :returns: The digest in lower case, or ``None`` if there is none.
    """
    tokens = data.decode("utf-8-sig", errors="replace").split()
    if tokens and _MD5_RE.match(tokens[0]):
        return tokens[0].lower()
    return None


def pack_validity(valid: np.ndarray) -> bytes:
    """Pack a validity mask into the bit array of a binary validity file.

    Bit ``j`` of the mask is bit ``j % 8`` of byte ``j // 8``, counted from
    the least significant bit.
    """
    return np.packbits(valid.ravel(), bitorder="little").tobytes()


def unpack_validity(data: bytes, count: int) -> np.ndarray:
    """Unpack the first ``count`` bits of a binary validity file into a boolean mask."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), count=count, bitorder="little")
    return bits.astype(np.bool_)


class PackedMember:
    """The still compressed content of a member of a read container.

    It holds only the compressed bytes and unpacks them each time :meth:`read` is
    called, so it is immutable and may be used from several threads.
    """

    __slots__ = ("_crc", "_deflated", "_name", "_raw", "_size")

    def __init__(self, name: str, raw: bytes, *, deflated: bool, size: int, crc: int) -> None:
        """Keep the compressed ``raw`` bytes of the member ``name``.

        :param name: The name of the member, for error messages.
        :param raw: The bytes as stored in the archive.
        :param deflated: Whether ``raw`` is deflated, otherwise it is stored as it is.
        :param size: The uncompressed size declared in the archive.
        :param crc: The CRC-32 of the uncompressed content declared in the archive.
        """
        self._name = name
        self._raw = raw
        self._deflated = deflated
        self._size = size
        self._crc = crc

    def read(self) -> bytes:
        """Return the uncompressed content.

        :raises X3pFormatError: If the member is damaged.
        """
        if self._deflated:
            decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                # One byte more than declared shows a member that is larger than declared.
                data = decompressor.decompress(self._raw, self._size + 1)
            except zlib.error as error:
                raise self._damaged(str(error)) from error
            if not decompressor.eof:
                raise self._damaged("the compressed data does not match the declared size")
        else:
            data = self._raw
        if len(data) != self._size:
            raise self._damaged("the size does not match the declared size")
        if zlib.crc32(data) != self._crc:
            raise self._damaged("bad CRC-32")
        return data

    def _damaged(self, reason: str) -> X3pFormatError:
        return X3pFormatError(f"Cannot read {self._name!r} from the container: {reason}")


class Container:
    """A read x3p zip container.

    Members are addressed by their path relative to the directory that holds
    ``main.xml``, normally the root of the zip. Directory entries and the
    junk some operating systems add are not members.

    :param data: The content of the file, or a seekable binary stream that holds it, from which
        only what is asked for is read.
    :raises X3pFormatError: If ``data`` is not a zip archive, or it holds no
        unambiguous ``main.xml``.
    """

    def __init__(self, data: bytes | IO[bytes]) -> None:
        """Open the archive and locate ``main.xml``."""
        try:
            self._zip = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data)
        except zipfile.BadZipFile:
            raise X3pFormatError("Not an x3p file, it is not a zip archive") from None

        entries: dict[str, zipfile.ZipInfo] = {}
        for info in self._zip.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or name.endswith("/"):
                continue
            if _is_junk(name):
                continue
            entries[name] = info

        prefix = self._locate_main(entries)
        self._members = {
            name[len(prefix) :]: info for name, info in entries.items() if name.startswith(prefix)
        }

    @staticmethod
    def _locate_main(entries: dict[str, zipfile.ZipInfo]) -> str:
        if MAIN_XML in entries:
            return ""
        nested = [
            name for name in entries if name.endswith("/" + MAIN_XML) and name.count("/") == 1
        ]
        if len(nested) == 1:
            return nested[0][: -len(MAIN_XML)]
        if nested:
            raise X3pFormatError("The container holds several folders with a main.xml")
        raise X3pFormatError(f"The container holds no {MAIN_XML}")

    @property
    def names(self) -> list[str]:
        """Paths of all members, relative to the folder of ``main.xml``."""
        return list(self._members)

    def has(self, name: str) -> bool:
        """Return whether the container holds the member ``name``."""
        return name in self._members

    def size(self, name: str) -> int:
        """Return the uncompressed size of a member as declared in the archive."""
        return self._members[name].file_size

    def read(self, name: str) -> bytes:
        """Return the content of a member.

        :raises X3pFormatError: If the member is damaged or encrypted.
        """
        return self._read(self._members[name])

    def pack(self, name: str) -> bytes | PackedMember:
        """Return a member without unpacking it, if it is stored or deflated.

        Other members, such as encrypted ones, are read immediately.

        :raises X3pFormatError: If such a member is damaged or encrypted.
        """
        info = self._members[name]
        if info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) and not (
            info.flag_bits & _ENCRYPTED
        ):
            raw = self._raw(info)
            if raw is not None:
                return PackedMember(
                    info.filename,
                    raw,
                    deflated=info.compress_type == zipfile.ZIP_DEFLATED,
                    size=info.file_size,
                    crc=info.CRC,
                )
        return self._read(info)

    def _raw(self, info: zipfile.ZipInfo) -> bytes | None:
        """Return the bytes of a member as stored, or ``None`` if the archive is odd."""
        stream = self._zip.fp
        if stream is None:
            return None
        stream.seek(info.header_offset)
        header = stream.read(_LOCAL_HEADER.size)
        if len(header) != _LOCAL_HEADER.size:
            return None
        fields = _LOCAL_HEADER.unpack(header)
        if fields[0] != _LOCAL_HEADER_SIGNATURE:
            return None
        stream.seek(info.header_offset + _LOCAL_HEADER.size + fields[-2] + fields[-1])
        raw = stream.read(info.compress_size)
        return raw if len(raw) == info.compress_size else None

    def _read(self, info: zipfile.ZipInfo) -> bytes:
        try:
            return self._zip.read(info)
        except (zipfile.BadZipFile, zlib.error, NotImplementedError, RuntimeError) as error:
            raise X3pFormatError(
                f"Cannot read {info.filename!r} from the container: {error}"
            ) from error


def write_container(members: Iterable[tuple[str, bytes]]) -> bytes:
    """Build the zip container of an x3p file.

    :param members: ``(path, content)`` pairs, written in the given order. The
        path is stored exactly as given, backslashes included, on every
        platform: the standard names the file of a vendor extension with them.
    :returns: The zip archive; identical members give identical bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            info = zipfile.ZipInfo("member", date_time=_ZIP_EPOCH)
            # Set after construction: ZipInfo would convert a backslash to a slash on Windows.
            info.filename = name
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    return buffer.getvalue()
