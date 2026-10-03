# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read a profile, with several layers and along a path."""

import numpy as np

import x3pio

# A stylus instrument is a contacting probing system; the identification says which stylus.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Stylus 100", "98765", "1.4"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.CONTACTING, "diamond stylus, 2 um tip"),
)

# A line scan at equal steps: the heights and the step along the line.
position = np.linspace(0.0, 4.0, 200)
scan = x3pio.Profile.from_array(1e-6 * np.sin(position), x_scale=20e-6, metadata=metadata)
print(type(scan).__name__, scan.layers[0].z.shape)  # (points,)
print("first positions along the line:", scan.layers[0].x_values[:3])

# Several scans of the same line are the layers of one profile.
scans = [1e-6 * np.sin(position + shift) for shift in (0.0, 0.1, 0.2)]
layered = x3pio.Profile.from_layers(scans, x_scale=20e-6, metadata=metadata)
print(len(layered.layers), "layers of", layered.layers[0].z.shape[0], "points")
for index, layer in enumerate(layered.layers):
    print(" layer", index, "peak", float(layer.z.max()))

# A profile does not have to lie on a straight line. Its points are then given by their
# coordinates, in the order along the profile; the same shape comes out as ``layer.points()``.
turn = np.linspace(0.0, np.pi, 50)
path = np.column_stack([np.cos(turn), np.sin(turn), 1e-3 * np.cos(4 * turn)]) * 1e-3
curved = x3pio.Profile.from_points(path, metadata=metadata)
print(
    "a curved profile:",
    curved.layers[0].points().shape,
    "axes absolute:",
    not curved.header.x.is_incremental,
)
