# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Vendor specific extensions of an x3p file, registered by ``VendorSpecificID``.

A vendor can put files of its own into the zip archive of an x3p file, such as the
settings of the instrument or notes on the measurement. A ``VendorSpecificID`` in ``main.xml``
is a URI that the vendor owns and that identifies such an extension, so that two vendors do not
take the same name. Software that does not know an ID ignores it and its files, and a file
that no ID registers is not an extension.

The two revisions of the standard differ in what an ID stands for. In ISO 25178-72:2017 there is a
single ID for the extension of a vendor as a whole, and its files lie in the container under any
name. Its Amd 1:2020 gives every file an ID of its own and derives the name of the file in the
container from the ID. :class:`VendorExtensions` holds both, as one mapping from an ID to the
content of its file; see :class:`~x3pio.Revision` for what differs.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from typing import Protocol, TypeAlias

from ..exceptions import X3pFormatError
from .header import DEFAULT_REVISION, Revision
from .naming import is_reserved, local_member_path, split_uri
from .streams import PathOrStream, read_bytes, write_bytes

__all__ = ["VendorExtensions", "convert_extensions"]


class PackedContent(Protocol):
    """Content that is not unpacked yet, such as a compressed member of a read container."""

    def read(self) -> bytes:
        """Return the content, unpacking it."""
        ...


#: What is kept for an extension file: its content, or the content still packed.
Stored: TypeAlias = bytes | PackedContent


def content_of(stored: Stored) -> bytes:
    """Return the content of a stored extension file, unpacking it if needed."""
    return stored if isinstance(stored, bytes) else stored.read()


#: The host name and the leading path segments that a view of the extensions is restricted to.
_Prefix = tuple[str, tuple[str, ...]]


class _Store:
    """The registrations and the files; shared by a mapping and its views."""

    def __init__(self, revision: Revision) -> None:
        self.revision = revision
        #: The one ``VendorSpecificID`` of the revision ``ISO5436_2000``, as given.
        self.vendor: str | None = None
        #: Every ID, with the name of its file in the container.
        self.ids: dict[str, str | None] = {}
        #: The files by their names in the container; the IDs that name the same file share one.
        self.files: dict[str, Stored] = {}


def _base(vendor: str) -> str:
    """Return a vendor without the whitespace around it and without a trailing ``/``."""
    text = vendor.strip().rstrip("/")
    if not text:
        raise X3pFormatError("A vendor must not be empty")
    return text


class VendorExtensions(MutableMapping[str, bytes | None]):
    """The vendor specific extension files of an x3p file, by their ``VendorSpecificID``.

    The standard registers vendor files by a ``VendorSpecificID``, a URI that the vendor owns:
    a domain name, an optional path and the name of the file. This is a mutable mapping from
    the IDs, exactly as written in the file, with the content of their files as values. The value is
    ``None`` for an ID that has no file, such as an ID that is only a domain name. The key
    is never changed, so give the ID as the software that reads the file expects to find it.

    Files are added with :meth:`add`, which takes the vendor and the path of the file below it
    and makes the ID from them, or with ``extensions[uri] = content`` for an ID that is known
    in full.

    >>> import numpy as np
    >>> import x3pio
    >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
    >>> uri = "http://www.vendor.com/mypath/a.xml"
    >>> file.extensions[uri] = b"<a/>"
    >>> file.extensions.add("http://www.vendor.com", "b.xml", b"<b/>")
    >>> list(file.extensions)
    ['http://www.vendor.com/mypath/a.xml', 'http://www.vendor.com/b.xml']
    >>> file.extensions[uri]
    b'<a/>'
    >>> list(file.extensions.below("http://www.vendor.com/mypath"))
    ['http://www.vendor.com/mypath/a.xml']
    >>> del file.extensions[uri]
    >>> len(file.extensions)
    1

    The two revisions hold them differently. In Amd 1:2020 every file has an ID of its own,
    and any number of vendors can add files. ISO 25178-72:2017 has a single
    ``VendorSpecificID``, which identifies the extension of the vendor as a whole, and its
    files lie in the container under any name. That ID is the vendor of the first file that
    is added, a file of another vendor is refused, and the key of a file is the vendor, a ``/``
    and its path.

    Different URIs can name the same file in the container, such as ``http://a.b/c`` and
    ``https://a.b/c``, because the name leaves out the scheme. They share its content. Adding a
    URI for a file that another ID names, with other content, raises an error.

    The files of a file that was read stay compressed until their content is requested, so
    listing the IDs unpacks nothing. A damaged file raises :class:`~x3pio.X3pFormatError` when
    its content is requested, not when the x3p file is read.
    """

    def __init__(self, revision: Revision = DEFAULT_REVISION) -> None:
        """Create an empty mapping for a revision."""
        self._store = _Store(revision)
        self._prefixes: tuple[_Prefix, ...] = ()

    @property
    def _revision(self) -> Revision:
        return self._store.revision

    @property
    def _has_ids(self) -> bool:
        """Whether every file has an ID of its own, as in Amd 1:2020."""
        return self._store.revision.has_vendor_ids

    # --- reading --------------------------------------------------------------------------

    def _matches(self, uri: str) -> bool:
        """Return whether ``uri`` is one of this mapping, or lies below what it is limited to."""
        if not self._prefixes:
            return True
        try:
            host, segments = split_uri(uri)
        except X3pFormatError:
            return False
        return all(
            host == prefix_host and tuple(segments[: len(prefix)]) == prefix
            for prefix_host, prefix in self._prefixes
        )

    def __getitem__(self, uri: str) -> bytes | None:
        """Return the content of the file of an ID, or ``None`` for an ID that has no file."""
        if uri not in self:
            raise KeyError(uri)
        member = self._store.ids[uri]
        return None if member is None else content_of(self._store.files[member])

    def __contains__(self, uri: object) -> bool:
        """Return whether ``uri`` is an ID of this mapping."""
        return isinstance(uri, str) and uri in self._store.ids and self._matches(uri)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the IDs, in the order of the file."""
        return iter([uri for uri in self._store.ids if self._matches(uri)])

    def __len__(self) -> int:
        """Return the number of IDs."""
        return sum(1 for uri in self._store.ids if self._matches(uri))

    def __eq__(self, other: object) -> bool:
        """Compare the IDs and the content of their files, and in a revision the vendor."""
        if isinstance(other, VendorExtensions):
            same_vendor = (self._store.vendor is None) == (other._store.vendor is None) and (
                self._store.vendor is None
                or other._store.vendor is None
                or _base(self._store.vendor) == _base(other._store.vendor)
            )
            return (
                self._revision is other._revision
                and same_vendor
                and dict(self.items()) == dict(other.items())
            )
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        """Show the IDs."""
        return f"VendorExtensions({list(self)!r})"

    def _copy(self) -> VendorExtensions:
        """Return an independent mapping of all the IDs and files, the whole of this one."""
        copy = VendorExtensions(self._revision)
        copy._store.vendor = self._store.vendor
        copy._store.ids = dict(self._store.ids)
        copy._store.files = dict(self._store.files)
        return copy

    @property
    def vendor_ids(self) -> list[str]:
        """The ``VendorSpecificID`` elements that a file written with these extensions has.

        In Amd 1:2020 there is one for every file, the same as the keys of the mapping. In ISO
        25178-72:2017 there is at most one, which identifies the extension of the vendor as a whole.

        >>> import x3pio
        >>> file = x3pio.Surface.from_array(
        ...     [[0.0]], x_scale=1.0, y_scale=1.0, revision=x3pio.Revision.ISO5436_2000
        ... )
        >>> file.extensions.vendor_ids
        []
        >>> file.extensions.add("http://www.vendor.com", "a.xml", b"<a/>")
        >>> file.extensions.vendor_ids
        ['http://www.vendor.com']
        """
        return self._parts()[0]

    def _has_file(self, uri: str) -> bool:
        """Return whether ``uri`` is registered with a file, without unpacking it."""
        return uri in self and self._store.ids[uri] is not None

    def extract(self, uri: str, destination: PathOrStream) -> None:
        """Write the content of a file to a file or a binary stream.

        :param uri: The ID of the file.
        :param destination: The path of the file to write, or an open binary stream.
        :raises KeyError: If ``uri`` is not registered.
        :raises X3pFormatError: If ``uri`` is registered without a file, or the file is
            damaged.
        :raises OSError: If the file cannot be written.
        """
        content = self[uri]
        if content is None:
            raise X3pFormatError(f"{uri!r} is registered without a file")
        write_bytes(destination, content)

    def below(self, prefix: str) -> VendorExtensions:
        """Return a live view of the IDs that lie below a domain name or a path.

        The view has the same keys. Reading, adding and removing through it changes this
        mapping, and a new ID has to lie below the prefix. ``prefix`` is matched against
        whole segments: the host name, which is everything up to the first ``/``, and then
        the segments of the path. The scheme is ignored, so ``http`` and ``https`` IDs
        of one vendor are both found.

        :param prefix: A URI of a vendor, such as ``"http://www.vendor.com"``, or of a
            directory of it, such as ``"http://www.vendor.com/mypath"``.
        :raises X3pFormatError: If ``prefix`` is not such a URI.

        >>> import numpy as np
        >>> import x3pio
        >>> file = x3pio.Surface.from_array(np.zeros((1, 2)), x_scale=1.0, y_scale=1.0)
        >>> file.extensions["http://www.vendor.com/mypath/a.xml"] = b"<a/>"
        >>> file.extensions["https://www.vendor.com/b.xml"] = b"<b/>"
        >>> file.extensions["http://example.org/c.xml"] = b"<c/>"
        >>> len(file.extensions.below("www.vendor.com"))
        2
        >>> file.extensions.below("www.vendor.com").clear()
        >>> list(file.extensions)
        ['http://example.org/c.xml']
        """
        host, segments = split_uri(prefix)
        view = VendorExtensions(self._revision)
        view._store = self._store
        view._prefixes = (*self._prefixes, (host, tuple(segments)))
        return view

    # --- inserting and removing -----------------------------------------------------------

    def __setitem__(self, uri: str, content: bytes | None) -> None:
        """Register the file of an ID, or replace its content; see :meth:`add`."""
        if content is None:
            raise TypeError("An extension file needs content; delete the ID to remove it")
        self._put_uri(uri, bytes(content))

    def add(
        self, vendor: str, path: str, content: bytes | bytearray | memoryview | PathOrStream
    ) -> None:
        """Add a file of a vendor, or replace the content of the one at ``path``.

        In Amd 1:2020 the ID of the file is ``vendor``, a ``/`` and ``path``. In the
        standard before it ``vendor`` is the single ``VendorSpecificID`` of the file and
        ``path`` is the path of the file in the container. The first file that is added fixes
        the vendor, and a file of another vendor is refused; the key of the file is again
        ``vendor``, a ``/`` and ``path``. A ``/`` at the end of ``vendor`` does not matter.

        :param vendor: The vendor, a domain name or a URI that the vendor owns, usually with
            a scheme such as ``http://``. It is written to the file exactly as given.
        :param path: The path of the file below the vendor, such as ``"mypath/a.xml"``.
        :param content: The content of the file: its ``bytes``, the path of a file to read
            (a ``str`` is always a path, never data), or an open binary stream, which is
            read from its current position and left open. The content is read immediately.
        :raises X3pFormatError: If ``vendor`` or ``path`` cannot name a file, the file would
            take the name of a file of the format, is both a file and a directory, or
            names the same file as another ID with other content, the vendor is another
            than the one of the file, or the ID does not lie below what this view is
            limited to.
        :raises OSError: If a file cannot be read.
        """
        self._check_target(vendor, path)
        data = (
            bytes(content)
            if isinstance(content, (bytes, bytearray, memoryview))
            else read_bytes(content)
        )
        name = local_member_path(path)
        if self._has_ids:
            self._insert(f"{_base(vendor)}/{name}", data)
        else:
            self._insert_file(vendor, name, data)

    def _put_uri(self, uri: str, content: bytes) -> None:
        """Add a file by the ID that it is known by."""
        if self._has_ids:
            self._insert(uri, content)
            return
        vendor = self._store.vendor
        if vendor is None:
            raise X3pFormatError(
                "The standard has a single VendorSpecificID, which the first file fixes; "
                "add a file with add(vendor, path, content) first"
            )
        text = uri.strip()
        base = _base(vendor)
        if not text.startswith(base + "/"):
            raise X3pFormatError(
                f"{uri!r} does not lie below the vendor {vendor!r}; "
                "the standard has a single vendor"
            )
        self._insert_file(vendor, text[len(base) + 1 :], content)

    def _check_target(self, vendor: str, path: str) -> None:
        """Check that a file of ``vendor`` can be added at ``path``, without changing anything."""
        name = local_member_path(path)
        if self._has_ids:
            self._member(f"{_base(vendor)}/{name}")
        else:
            self._check_file(vendor, name)

    def _member(self, uri: str) -> str:
        """Check that ``uri`` can be registered here, and return the name of its file."""
        host, segments = split_uri(uri, registering=True)
        if not segments:
            raise X3pFormatError(f"{uri!r} needs the path to a file after the domain name")
        if not self._matches(uri.strip()):
            raise X3pFormatError(f"{uri!r} does not lie below what this view is limited to")
        member = "\\".join([*host.split("."), *segments])
        if is_reserved(member):
            raise X3pFormatError(f"{uri!r} would take the name of a file of the format")
        return member

    def _insert(self, uri: str, content: bytes) -> None:
        member = self._member(uri)
        uri = uri.strip()
        store = self._store
        if member in store.files and store.ids.get(uri) != member:
            if content_of(store.files[member]) != content:
                other = next(known for known, name in store.ids.items() if name == member)
                raise X3pFormatError(
                    f"{uri!r} names the same file in the container as {other!r}, which has "
                    "other content; remove that ID first, or give the same content"
                )
        elif member not in store.files:
            for name in store.files:
                if name.startswith(member + "\\") or member.startswith(name + "\\"):
                    raise X3pFormatError(f"{uri!r} cannot be both a file and a directory")
        store.files[member] = content
        store.ids[uri] = member

    def _check_file(self, vendor: str, name: str) -> str:
        """Check that a file can be added at ``name`` for ``vendor``, and return its key."""
        base = _base(vendor)
        current = self._store.vendor
        if current is not None and _base(current) != base:
            raise X3pFormatError(
                f"The standard has a single vendor, {current!r}; {vendor!r} is another one"
            )
        key = f"{base}/{name}"
        if is_reserved(name):
            raise X3pFormatError(f"{key!r} would take the name of a file of the format")
        if not self._matches(key):
            raise X3pFormatError(f"{key!r} does not lie below what this view is limited to")
        if name not in self._store.files:
            for known in self._store.files:
                if known.startswith(name + "/") or name.startswith(known + "/"):
                    raise X3pFormatError(f"{key!r} cannot be both a file and a directory")
        return key

    def _insert_file(self, vendor: str, name: str, content: bytes) -> None:
        key = self._check_file(vendor, local_member_path(name))
        name = local_member_path(name)
        if self._store.vendor is None:
            self._store.vendor = vendor.strip()
        self._store.files[name] = content
        self._store.ids[key] = name

    def __delitem__(self, uri: str) -> None:
        """Remove an ID, and its file if no other ID names it."""
        if uri not in self:
            raise KeyError(uri)
        store = self._store
        member = store.ids.pop(uri)
        if member is not None and member not in store.ids.values():
            del store.files[member]

    def clear(self) -> None:
        """Remove every ID of this mapping, or of this view, and the files that no ID names.

        Clearing the whole mapping also removes the vendor of ISO 25178-72:2017.
        """
        for uri in list(self):
            del self[uri]
        if not self._prefixes:
            self._store.vendor = None

    # --- reading and writing the container ------------------------------------------------

    def _announce(self, uri: str, stored: Stored | None, member: str | None) -> None:
        """Register an ID read from ``main.xml``, with its file if the container holds one."""
        if stored is None or member is None:
            self._store.ids[uri] = None
            return
        self._store.ids[uri] = member
        self._store.files.setdefault(member, stored)

    @classmethod
    def _read_files(cls, vendor: str, files: Mapping[str, Stored]) -> VendorExtensions:
        """Build the extensions of ISO 25178-72:2017 from what a file holds."""
        extensions = cls(Revision.ISO5436_2000)
        extensions._store.vendor = vendor
        base = vendor.strip().rstrip("/")
        for name, stored in files.items():
            extensions._store.files[name] = stored
            extensions._store.ids[f"{base}/{name}"] = name
        return extensions

    def _parts(self) -> tuple[list[str], dict[str, Stored]]:
        """Return the IDs to write and the files to write, by their names in the container."""
        if self._has_ids:
            return list(self._store.ids), dict(self._store.files)
        vendor = self._store.vendor
        return ([vendor.strip()] if vendor else []), dict(self._store.files)


def convert_extensions(
    extensions: VendorExtensions, revision: Revision, *, drop: bool = False
) -> VendorExtensions:
    """Convert extensions to the other revision.

    Amd 1:2020 gives every file an ID of its own, and ISO 25178-72:2017 has a single
    ID for the extension as a whole.

    - To Amd 1:2020, each file becomes the extension with the ID that is the vendor, a ``/``
      and the path of the file, and a vendor without files stays an ID that names no file.
    - To ISO 25178-72:2017, the extensions cannot be converted and have to be dropped.

    :param drop: Leave the extensions out when converting to ISO 25178-72:2017,
        instead of failing.
    :raises X3pFormatError: If there are extensions and they would have to be dropped.
    """
    if revision.has_vendor_ids:
        return _to_ids(extensions)
    if extensions and not drop:
        raise X3pFormatError(
            "The extensions cannot be converted to ISO 25178-72:2017, since "
            "an ID does not say where the vendor ends and the path of a file begins; "
            "drop them to convert the file anyway"
        )
    return VendorExtensions(Revision.ISO5436_2000)


def _to_ids(extensions: VendorExtensions) -> VendorExtensions:
    target = VendorExtensions(Revision.ISO25178_72_2017_DAM1)
    store = extensions._store
    if store.vendor is not None and not store.ids:
        target._announce(store.vendor.strip(), None, None)
    for uri in extensions:
        content = extensions[uri]  # every ID of the standard has a file
        try:
            if content is not None:
                target._insert(uri, content)
        except X3pFormatError as error:
            raise X3pFormatError(
                f"The file of {uri!r} cannot get an ID of its own from {store.vendor!r}: {error}"
            ) from error
    return target
