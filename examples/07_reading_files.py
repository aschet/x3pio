# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Read a file of unknown kind."""

import io
import zipfile

import numpy as np

import x3pio

# Make one file of each kind in memory; ``read`` takes a path or an open file as well.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Scanner 3000", "12345", "2.1"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.NON_CONTACTING, "confocal"),
)
files = [
    x3pio.Surface.from_array(np.zeros((6, 8)), x_scale=1e-6, y_scale=1e-6, metadata=metadata),
    x3pio.Surface.from_layers([np.zeros((3, 4))] * 3, x_scale=1e-6, y_scale=1e-6),
    x3pio.Profile.from_array(np.zeros(100), x_scale=1e-6, metadata=metadata),
    x3pio.PointCloud.from_points(np.random.default_rng(0).uniform(0, 1e-3, (250, 3))),
]

for original in files:
    x3p = x3pio.read(io.BytesIO(original.dumps()))  # a Profile, a Surface or a PointCloud

    # What all kinds share: layers with points, the placement, the metadata and the revision.
    shapes = [layer.points().shape for layer in x3p.layers]
    print(f"{type(x3p).__name__:10}", shapes, x3p.revision.name)

    # The metadata is optional, so a file may not have any. A field may be empty.
    if x3p.metadata is not None:
        print(
            "           ", x3p.metadata.instrument.model or "(no model)", "|", x3p.metadata.creator
        )

    # What belongs to a kind is found by asking its type.
    match x3p:
        case x3pio.Surface() | x3pio.Profile():
            print("           ", len(x3p.layers), "layer(s) of", x3p.layers[0].z.shape)
        case x3pio.PointCloud():
            print("            an unordered list of", len(x3p.layers[0].z), "points")

# To look at many files, ``info`` reads the description of a file and leaves its data alone.
summary = x3pio.info(io.BytesIO(files[1].dumps()))
print(summary.feature_type.name, summary.layer_count, "layers of", summary.shape)

# A file of another kind than the one that was asked for is an error.
try:
    x3pio.Surface.loads(files[3].dumps())
except x3pio.X3pFormatError as error:
    print("asking for a surface:", error)

# The MD5 checksums of a file are verified. A file that was changed after it was written raises
# ``X3pChecksumError``, and ``verify=False`` skips the check. Both errors derive from ``X3pError``.
source = zipfile.ZipFile(io.BytesIO(files[0].dumps()))
changed = io.BytesIO()
with zipfile.ZipFile(changed, "w") as target:
    for name in source.namelist():
        content = source.read(name)
        target.writestr(name, content + b"\n" if name == "main.xml" else content)
try:
    x3pio.read(io.BytesIO(changed.getvalue()))
except x3pio.X3pChecksumError as error:
    print("changed after it was written:", error)
unchecked = x3pio.read(io.BytesIO(changed.getvalue()), verify=False)
print("without the check:", unchecked.layers[0].z.shape)
