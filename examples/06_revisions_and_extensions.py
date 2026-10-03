# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read a file in each revision, with vendor extensions."""

from datetime import UTC, datetime

import numpy as np

import x3pio
from x3pio import Revision

# The metadata of a measurement. ISO 25178-72:2017 requires the date of the last calibration,
# Amd 1:2020 makes it optional: give it if it is known, and a file in the first revision gets
# the date of creation otherwise.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Scanner 3000", "12345", "2.1"),
    calibration_date=datetime(2026, 1, 15, 9, 30, tzinfo=UTC),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.NON_CONTACTING, "confocal"),
)

# A new file follows Amd 1:2020. ``revision`` writes ISO 25178-72:2017.
surface = x3pio.Surface.from_array(np.zeros((4, 4)), x_scale=1e-6, y_scale=1e-6, metadata=metadata)
print(surface.revision.name, "|", surface.revision.value)

# A vendor adds files to the container, each registered by a URI that the vendor owns.
# ``add`` takes the vendor and the path of the file below it.
notes = b"Measured on the synthetic bench.\n"
surface.extensions.add("http://www.acme.example", "notes/sample.txt", notes)
surface.extensions["http://www.acme.example/settings.xml"] = b"<settings/>"
for uri in surface.extensions:  # the IDs, as the file has them
    print(" ", uri, len(surface.extensions[uri]), "bytes")

# ISO 25178-72:2017 has a single ID for the extension of a vendor, and its files lie in the
# container under any name. The same calls work, for one vendor. Extensions are not converted to
# this revision; ``drop_extensions`` drops them.
older = surface.with_revision(Revision.ISO5436_2000, drop_extensions=True)
older.extensions.add("http://www.acme.example", "notes/sample.txt", notes)
older.extensions.add("http://www.acme.example", "settings.xml", b"<settings/>")
print(older.revision.name, list(older.extensions))
try:
    older.extensions.add("http://other.example", "x.dat", b"1")
except x3pio.X3pFormatError as error:
    print("another vendor:", error)

# Converting back to Amd 1:2020 keeps every file: the ID of a file is the vendor and its path.
again = older.with_revision(Revision.ISO25178_72_2017_DAM1)
print(sorted(again.extensions) == sorted(surface.extensions))

# The content of a file is requested by its ID. In a file that is read, it stays compressed until
# then.
print(len(again.extensions["http://www.acme.example/notes/sample.txt"]), "bytes")
