# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read the metadata of a measurement."""

from datetime import UTC, datetime

import numpy as np

import x3pio
from x3pio import Instrument, Metadata, ProbingSystem, ProbingType

# The metadata is optional, but the standard asks for a creation date and a probing type.
# A new file gets a record that names x3pio as the software and the time of creation.
surface = x3pio.Surface.from_array(np.zeros((8, 8)), x_scale=1e-6, y_scale=1e-6)
print(surface.metadata.instrument.manufacturer, surface.metadata.probing_system.type.name)

# ``with_metadata`` makes a copy with the fields that are known; the file itself is not changed.
surface = surface.with_metadata(
    creator="Jane Doe",
    manufacturer="Acme Metrology",
    model="Scanner 3000",
    serial="12345",
    version="firmware 2.1",
    probing_type=ProbingType.NON_CONTACTING,
    probing_identification="confocal, 50x objective",
    calibration_date=datetime(2026, 1, 15, 9, 30, tzinfo=UTC),
    comment="Reference artefact",
)
print(surface.metadata.probing_system)

# Or build the whole record, and give it to a new file. Without a ``date``, the file gets the time
# of its creation; give the time of the measurement if it is known.
metadata = Metadata(
    date=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    instrument=Instrument(manufacturer="Acme Metrology", model="Scanner 3000", version="2.1"),
    probing_system=ProbingSystem(ProbingType.CONTACTING, "stylus, 2 um tip"),
)
stylus = x3pio.Profile.from_array(np.zeros(10), x_scale=1e-6, metadata=metadata)
print(stylus.metadata.probing_system.identification)

# ``with_metadata`` makes a changed copy and leaves the file as it is.
renamed = stylus.with_metadata(comment="second run")
print(renamed.metadata.comment, "|", stylus.metadata.comment)

# Data that a program made from other data is not measured. ``for_software`` makes the record for
# it: the software is the instrument, and the probing system is of the kind ``SOFTWARE``.
filtered = x3pio.Profile.from_array(
    np.zeros(10), x_scale=1e-6, metadata=Metadata.for_software("Profile Filter", "1.2")
)
print(filtered.metadata.instrument.model, filtered.metadata.probing_system.type.name)

# A date without a time zone is a local time, as ISO 8601 says; a date with one is kept as it is.
print(stylus.metadata.date.isoformat())
