# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Serialization of the model: zip container, ``main.xml``, binary files and checksums.

This package turns bytes into the types of ``x3pio.model`` and back. It depends on the model,
never the other way round.

When reading, x3pio tolerates the deviations from the standard that are listed here and
corrects them silently. Other deviations raise :class:`~x3pio.X3pFormatError`. Elements that
the schema does not define are ignored wherever they occur.

Reading does not validate ``main.xml`` against the XSD schema of the standard: the elements are
looked up by name and checked as far as the model needs. Writing always follows the standard. That
is ensured by tests that validate the written ``main.xml`` against the schemas of ISO
25178-72:2017 and Amd 1:2020.

Checksums. The MD5 checksums of ``main.xml``, of the binary data file and of the validity file
are verified. A mismatch raises :class:`~x3pio.X3pChecksumError`: the file was changed or damaged
after it was written. ``read(path, verify=False)``, ``X3pFile.open`` and ``X3pFile.loads`` skip
the check. A checksum that is missing or malformed, a checksum file that is missing, outside the
container, with another name or with only the digest, in either case and with or without the
file name that ``md5sum`` writes, is accepted.

Axes:

- A missing ``Increment`` is read as ``1``, and a missing ``DataType`` of an incremental axis as
  ``float64``. ISO 25178-72:2017 allows both, Amd 1:2020 requires them, and some files and
  instruments omit the increment of the z axis. An empty ``Increment`` is read as ``1`` and an
  empty ``Offset`` as ``0``.
- A negative ``Increment`` is accepted, a zero one is an error.
- A rotation element slightly outside of ``[-1, 1]``, by up to ``1e-6``, is clamped.

Document structure:

- A ``Revision`` text other than that of Amd 1:2020 is read as ISO 25178-72:2017, see
  :class:`~x3pio.Revision`.
- A root element outside of the namespace of the schema is accepted.
- A missing ``Record4``, ``Instrument`` or ``ProbingSystem``, or a missing element of the
  instrument, is empty.
- A ``FeatureType`` that does not agree with the data is corrected: a list is a point cloud, a
  matrix is a surface, and a profile has a single row.
- Placeholder text instead of a date or a probing type (``N/A``, ``Date of Calibration``,
  ``Type``) is dropped, leaving ``None``. The probing type is matched without regard to case.
  Metadata without a date or probing type cannot be written until they are set.

Numbers:

- A ``Datum`` outside of the number pattern of the schema, such as ``2.e-06``, is parsed anyway.
- A floating point value of ``inf``, ``-inf`` or ``NaN`` in a binary file is a point that is not
  measured, although the standard only mentions ``NaN``.

Container:

- A UTF-8 byte order mark before the XML declaration is ignored.
- Files packed by hand, for example by compressing the folder of an extracted file in the file
  manager, have all their content in a single folder instead of the root of the zip. This is
  accepted. Such a container may also hold what the operating system adds: a ``__MACOSX/``
  folder at any depth, ``._name`` companions and ``.DS_Store`` of macOS, and ``Thumbs.db`` and
  ``desktop.ini`` of Windows, in any case. These are dropped; other names are not touched.
- Files that no ``VendorSpecificID`` registers are ignored. The ``VendorSpecificID`` of any file
  is kept as written, whether it names a file or not.

Some problems cannot be corrected and always raise :class:`~x3pio.X3pFormatError`: an incremental
z axis, a zero increment, an unknown feature, axis or data type, data that does not fit the
dimensions, a point cloud with an incremental axis, a missing ``main.xml``, ``Record1`` or
``Record3``, links that leave the container (absolute paths, ``..``, URLs), and anything that is
not XML.
"""
