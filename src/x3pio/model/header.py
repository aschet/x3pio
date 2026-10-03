# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Axes, revision and metadata of an x3p file.

The ``main.xml`` document of a file has four records. The first says how the three axes are
stored, here :class:`Header` of :class:`Axis`: whether each is incremental or absolute, with what
data type, and by which increment its stored numbers are multiplied to give metres. It also names
the revision of the standard that the file follows, its :class:`Revision`, which belongs to the
file as a whole. The second, here :class:`Metadata`,
is optional: when and by whom the data was made, with which instrument and probing system, and
when the instrument was last calibrated. Where the coordinates lie in space is the placement of
the file, :class:`~x3pio.Placement`. The third record holds the data, and the fourth the checksum
of the document; those are the business of the codec.

The standard asks for a creation date and a probing type in the metadata, so metadata without
them cannot be written. The probing system is one of three kinds: contacting, non contacting,
and software, for data that was made or changed by software and not measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from ..exceptions import X3pFormatError
from .datatypes import DataType

__all__ = [
    "Axis",
    "AxisType",
    "FeatureType",
    "Header",
    "Instrument",
    "Metadata",
    "ProbingSystem",
    "ProbingType",
    "Revision",
    "validate_increment",
]


#: The dashes that software writes for a hyphen: the hyphen, figure dash, en dash, em dash,
#: horizontal bar and minus sign of Unicode.
_DASHES = {ord(dash): "-" for dash in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"}


class Revision(StrEnum):
    """Revision of the standard a file follows, as embedded verbatim in its ``Revision`` element.

    Members are named after the standard; their string value is the exact text of the
    ``Revision`` element. The two revisions differ in how a file holds vendor specific
    extensions: ISO 25178-72:2017 has a single ``VendorSpecificID`` for the extension as a
    whole, and its files lie in the container under any name. Its Amd 1:2020 registers each
    extension file by an ID of its own. Both are held by :class:`~x3pio.VendorExtensions`.

    ``X3pFile.revision`` is the revision of a file. A file is written in the revision it was read
    in, and a new file follows Amd 1:2020 unless ``revision=`` says otherwise.

    - Reading: a ``Revision`` with the text of Amd 1:2020 means ``ISO25178_72_2017_DAM1``. Any
      other text, or none, means ``ISO5436_2000``. A file with several ``VendorSpecificID`` is
      read as Amd 1:2020, since only Amd 1:2020 allows that.
    - Writing: the revision is written as the ``Revision`` element and decides how vendor
      extensions are written. Axes always have ``DataType``, ``Increment`` and ``Offset``, which
      both schemas accept.
    - Calibration date: the text of the standard leaves the ``CalibrationDate`` out when there
      was no calibration, but its schema requires it, and the schema of Amd 1:2020 makes it
      optional. ``ISO25178_72_2017_DAM1`` leaves it out when ``calibration_date`` is ``None``,
      while ``ISO5436_2000`` writes the date of creation to be valid against its schema. This
      states a calibration that did not happen, so set the date if you know it. A file without
      metadata has no ``CalibrationDate``.
    - Coordinates: both revisions use the definition of Amd 1:2020. The ``Increment`` scales
      the stored values of an absolute axis, and the first point of an incremental axis is at its
      offset. The text of ISO 25178-72:2017 does not say where the index of an incremental axis
      starts.

    :meth:`~x3pio.X3pFile.with_revision` converts a file to the other revision.
    """

    #: ISO 25178-72:2017, the ``ISO 5436-2`` XML format of openGPS.
    ISO5436_2000 = "ISO5436 - 2000"
    #: ISO 25178-72:2017/Amd 1:2020.
    ISO25178_72_2017_DAM1 = "ISO25178-72:2017/DAM1"

    @classmethod
    def resolve(cls, revision: str) -> Revision:
        """Return the revision of the text of a ``Revision`` element.

        The text of Amd 1:2020 gives ``ISO25178_72_2017_DAM1``, also with another dash than the
        hyphen, such as the en dash that some software writes. Any other text, such as the
        ``ISO5436 - 2000`` of ISO 25178-72:2017 or a variation of it, gives ``ISO5436_2000``.
        """
        text = revision.strip().translate(_DASHES)
        return cls.ISO25178_72_2017_DAM1 if text == cls.ISO25178_72_2017_DAM1 else cls.ISO5436_2000

    @property
    def has_vendor_ids(self) -> bool:
        """Whether every vendor extension file has a ``VendorSpecificID`` of its own."""
        return self is Revision.ISO25178_72_2017_DAM1

    @property
    def requires_calibration_date(self) -> bool:
        """Whether the schema requires a ``CalibrationDate`` in the metadata.

        The text of the standard says the element is missing for an instrument that has not
        been calibrated, but its schema requires it; Amd 1:2020 makes it optional.
        """
        return self is Revision.ISO5436_2000


#: The revision that new files are written in.
DEFAULT_REVISION = Revision.ISO25178_72_2017_DAM1


class FeatureType(StrEnum):
    """Class of the 3-D data in a file, as written in the ``FeatureType`` element.

    A profile or a surface is stored as a matrix of points whose neighbours
    in the matrix are also neighbours in space; a point cloud is an
    unordered list of points without any topology.
    """

    PROFILE = "PRF"
    SURFACE = "SUR"
    POINT_CLOUD = "PCL"


class AxisType(StrEnum):
    """Kind of an axis, as written in the ``AxisType`` element.

    An incremental axis derives its coordinate from the matrix index and
    stores nothing; an absolute axis stores a coordinate for each point.
    """

    INCREMENTAL = "I"
    ABSOLUTE = "A"


class ProbingType(StrEnum):
    """Kind of probing system, as written in the ``ProbingSystem/Type`` element."""

    CONTACTING = "Contacting"
    NON_CONTACTING = "NonContacting"
    SOFTWARE = "Software"


def validate_increment(increment: float) -> None:
    """Reject an increment that is not a positive, finite number.

    An increment is a length in metres: it is the step between two points of an incremental
    axis, and the factor that turns the stored numbers of an absolute axis into metres. Zero
    would collapse the axis, and a negative value is not allowed.

    :param increment: The increment in metres.
    :raises X3pFormatError: If ``increment`` is not a finite number above zero.
    """
    if not math.isfinite(increment) or increment <= 0:
        raise X3pFormatError(f"Increment must be a positive number, got {increment!r}")


@dataclass(frozen=True)
class Axis:
    """Description of one coordinate axis (``CX``, ``CY`` or ``CZ``), an immutable value.

    A stored coordinate ``s`` (a point's matrix index for an incremental
    axis) gives the view coordinate ``increment * s`` in metres. Where the view system lies in
    the global one is the :class:`~x3pio.Placement` of the file, not a property of an axis.

    :param axis_type: Incremental or absolute; the z axis is always absolute.
    :param data_type: Storage type of the stored coordinates. Irrelevant for an
        incremental axis, which stores none.
    :param increment: Scale factor in metres: the sampling interval of an
        incremental axis, or the scale of the stored values of an absolute
        axis (``1`` to store metres directly).
    """

    axis_type: AxisType = AxisType.INCREMENTAL
    data_type: DataType = DataType.FLOAT64
    increment: float = 1.0

    @property
    def is_incremental(self) -> bool:
        """Whether the coordinate is derived from the matrix index."""
        return self.axis_type is AxisType.INCREMENTAL


def _default_z_axis() -> Axis:
    return Axis(axis_type=AxisType.ABSOLUTE)


@dataclass(frozen=True)
class Header:
    """The three axes of x3p (``Record1/Axes``), an immutable value.

    The layers of a file share one header. To change an axis, make another header with
    :func:`dataclasses.replace`, or use the methods of the file that do it. The revision of the
    standard is not part of it, since it belongs to the whole file: see
    :attr:`~x3pio.X3pFile.revision`.

    The class of the data is not part of it: it is the type of the file, see
    :class:`~x3pio.Profile`, :class:`~x3pio.Surface` and :class:`~x3pio.PointCloud`, and so is the
    rotation and the offset of the view coordinate system, see :class:`~x3pio.Placement`.

    :param x: Description of the x axis.
    :param y: Description of the y axis.
    :param z: Description of the z axis, which must be absolute.
    """

    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    z: Axis = field(default_factory=_default_z_axis)


@dataclass
class Instrument:
    """The instrument or software that created the data (``Record2/Instrument``).

    All elements are required by the standard; ``serial`` may be empty for
    software-created data.
    """

    manufacturer: str = ""
    model: str = ""
    serial: str = ""
    version: str = ""


@dataclass
class ProbingSystem:
    """The probing system used (``Record2/ProbingSystem``).

    :param type: Kind of probing system; ``ProbingType.SOFTWARE`` for
        synthetic or filtered data and software gauges. ``None`` if a file
        read held no recognizable value, which must be fixed before writing.
    :param identification: Description as specific as possible, e.g. the
        stylus or objective used.
    """

    type: ProbingType | None = ProbingType.SOFTWARE
    identification: str = ""


@dataclass
class Metadata:
    """``Record2`` of x3p: the optional metadata of a data set.

    :param date: Date and time of creation. It is written as ``YYYY-MM-DDThh:mm:ss.sTZD``, with a
        decimal fraction and a time zone offset such as ``+02:00``. A value without a time zone
        is a local time, as in ISO 8601, and is written with the offset of the local time zone;
        a date without a zone that is read from a file is taken the same way. ``None`` if a file
        read held no recognizable value, which must be fixed before writing; a record needs one
        to be written, and :meth:`for_software` sets the current UTC time.
    :param creator: Name of the person or institution that created the data.
    :param instrument: The instrument or software that created the data.
    :param calibration_date: Date and time of the last calibration, written like ``date``;
        ``None`` if there was no calibration. The schema of the revision
        ``ISO5436_2000`` requires the element, so a file in it is written with the ``date`` of
        creation if there is none, and :meth:`~x3pio.X3pFile.with_revision` sets it so.
    :param probing_system: The probing system used.
    :param comment: Free text describing the data set.
    """

    date: datetime | None = None
    creator: str | None = None
    instrument: Instrument = field(default_factory=Instrument)
    calibration_date: datetime | None = None
    probing_system: ProbingSystem = field(default_factory=ProbingSystem)
    comment: str | None = None

    @classmethod
    def for_software(cls, name: str, version: str, *, date: datetime | None = None) -> Metadata:
        """Return the record of data that software made, not a measurement.

        The software is both the manufacturer and the model of the instrument, without a serial
        number, and the probing system is of the kind ``ProbingType.SOFTWARE``. Set what else
        is known, such as the creator and a comment, with :meth:`~x3pio.X3pFile.with_metadata`.

        :param name: Name of the software.
        :param version: Version of the software.
        :param date: Date and time of creation; the current UTC time by default.

        >>> metadata = Metadata.for_software("analyser", "1.2")
        >>> metadata.instrument.model, metadata.probing_system.type.name
        ('analyser', 'SOFTWARE')
        """
        return cls(
            date=datetime.now(UTC) if date is None else date,
            instrument=Instrument(name, name, "", version),
            probing_system=ProbingSystem(ProbingType.SOFTWARE, name),
        )
