# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read a surface, with one layer and with several."""

import tempfile
from pathlib import Path

import numpy as np

import x3pio

# The metadata describes the measurement. The standard asks for a creation date, which a new
# file sets to now if the record has none, and for the kind of probing system.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Scanner 3000", "12345", "firmware 2.1"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.NON_CONTACTING, "confocal, 50x objective"),
    comment="Reference artefact",
)

# A surface on a regular grid: the heights in metres, and the distance between the
# points along x and y. NaN marks a point that was not measured.
heights = 1e-6 * np.random.default_rng(0).normal(size=(480, 640))
heights[100:110, 200:210] = np.nan

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "surface.x3p"
    x3pio.write(path, heights, x_scale=1e-6, y_scale=1e-6, metadata=metadata)

    surface = x3pio.Surface.open(
        path
    )  # the type is part of the call: another kind of file is an error
    print(type(surface).__name__, len(surface.layers), "layer")
    print("measured with", surface.metadata.instrument.model, "by", surface.metadata.creator)

    layer = surface.layers[0]  # the first layer; a file can have several
    print("heights of the layer:", layer.z.shape, "measured points:", int(layer.valid.sum()))

    # y grows with the row and x with the column; the axes are in metres.
    width, height = layer.x_values[-1] * 1e3, layer.y_values[-1] * 1e3
    print(f"the surface is {width:.3f} mm wide and {height:.3f} mm high")
    print("a regular grid:", surface.is_regular_grid)

# Several surfaces of the same size, such as repeated measurements, are the layers of one file.
repeats = x3pio.Surface.from_layers(
    [heights, heights + 1e-7, heights + 2e-7], x_scale=1e-6, y_scale=1e-6, metadata=metadata
)
print(
    len(repeats.layers), "layers; the points of the second one:", repeats.layers[1].points().shape
)

# The heights are stored as 64-bit floating point numbers, in metres. A 16-bit integer file is
# much smaller: its increment is the smallest one that fits the range of the heights.
compact = x3pio.Surface.from_array(
    heights, x_scale=1e-6, y_scale=1e-6, z_type=x3pio.DataType.INT16, metadata=metadata
)
print("increment of the integer file:", compact.header.z.increment)

# The coordinates are stored in a binary file by default, or as XML text.
with tempfile.TemporaryDirectory() as folder:
    for storage in x3pio.DataStorage:
        path = Path(folder) / f"{storage.name.lower()}.x3p"
        compact.save(path, storage=storage)
        print(storage.name.lower(), path.stat().st_size, "bytes")
