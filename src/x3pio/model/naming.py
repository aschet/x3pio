# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Names and paths of the members of an x3p container.

An x3p file is a zip archive. The standard names the files that the format itself uses, and it
derives the name of a vendor extension file from its ``VendorSpecificID``. These are string rules
that do not depend on how a zip archive is read or written, so the model uses them too.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from ..exceptions import X3pFormatError

__all__ = [
    "CHECKSUM_FILE",
    "DATA_FILE",
    "MAIN_XML",
    "RESERVED_NAMES",
    "VALID_FILE",
    "is_reserved",
    "local_member_path",
    "split_uri",
    "vendor_extension_path",
]

#: The central document of every container.
MAIN_XML = "main.xml"
#: Recommended name of the checksum file of ``main.xml``.
CHECKSUM_FILE = "md5checksum.hex"
#: Recommended path of the binary coordinate file.
DATA_FILE = "bindata/data.bin"
#: Recommended path of the binary validity file.
VALID_FILE = "bindata/valid.bin"
#: Members that a file written by :func:`write_container` always uses itself.
RESERVED_NAMES = frozenset({MAIN_XML, CHECKSUM_FILE, DATA_FILE, VALID_FILE})


#: A URI scheme, or the drive letter of a Windows path.
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def local_member_path(link: str) -> str:
    """Normalize a link to a container member, rejecting anything not local to the container.

    The standard requires links to point into the container: software must
    not follow a link to a network resource or to somewhere else on disk.
    Backslashes are accepted as path separators.

    :raises X3pFormatError: If ``link`` is empty, absolute, has a URI scheme or
        drive letter, or climbs out of the container with ``..``.
    """
    text = link.strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or text.startswith("/") or _SCHEME_RE.match(text) or ".." in parts:
        raise X3pFormatError(f"Link {link!r} does not point to a member of the container")
    return "/".join(parts)


def is_reserved(name: str) -> bool:
    """Return whether a member path collides with a file that is always written."""
    return local_member_path(name) in RESERVED_NAMES


def split_uri(uri: str, *, registering: bool = False) -> tuple[str, list[str]]:
    """Split a ``VendorSpecificID`` into its host name and the segments of its path.

    The scheme is not part of either, and may be missing. The host name is everything
    before the first ``/``, as in any URI. A path may be empty.

    >>> split_uri("http://www.vendor.com/mypath/myelements.xml")
    ('www.vendor.com', ['mypath', 'myelements.xml'])

    :param uri: The URI.
    :param registering: Also refuse what makes an awkward name in the container: a user name or
        a port in the host name.
    :raises X3pFormatError: If ``uri`` has no host name, an empty label of it, an empty or
        relative (``.`` or ``..``) segment, a backslash, a query or a fragment.
    """
    text = uri.strip()
    try:
        parts = urlsplit(text if "://" in text else "//" + text)
    except ValueError:
        raise X3pFormatError(f"{uri!r} is not a URI") from None
    host = parts.netloc
    segments = [] if parts.path in ("", "/") else parts.path.removeprefix("/").split("/")
    bad = (
        not host
        or any(label == "" for label in host.split("."))
        or any(segment in ("", ".", "..") for segment in segments)
        or any(character in text for character in "\\?#")
        or (registering and any(character in host for character in ":@"))
    )
    if bad:
        raise X3pFormatError(f"{uri!r} is not a URI of a domain name with a path")
    return host, segments


def vendor_extension_path(uri: str) -> str:
    r"""Return the container path of the file belonging to a ``VendorSpecificID`` URI.

    The standard places the vendor specific file under the path
    of its URI, with the separators of the domain name and the path replaced
    by backslashes, e.g. ``http://www.vendor.com/mypath/myelements.xml``
    becomes ``www\vendor\com\mypath\myelements.xml``.

    >>> vendor_extension_path("http://www.vendor.com/mypath/myelements.xml")
    'www\\vendor\\com\\mypath\\myelements.xml'

    :param uri: The ``VendorSpecificID``.
    :returns: The name of the file in the container, with ``\`` as separator.
    :raises X3pFormatError: If ``uri`` has no path naming a file, or contains
        an empty or relative path segment.
    """
    try:
        host, segments = split_uri(uri)
    except X3pFormatError:
        segments = []
    if not segments:
        raise X3pFormatError(f"VendorSpecificID {uri!r} must be a URI with a path to a file")
    return "\\".join([*host.split("."), *segments])
