# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read an irregular surface and a point cloud."""

import io

import numpy as np

import x3pio

# Data that a program made, such as a simulation or a filter, is not measured: its probing
# system is the software, and the comment says what was done.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Simulator", "", "0.3"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.SOFTWARE, "synthetic surface"),
    comment="Bent grid",
)

# A surface is a matrix of points that keep the neighbours of their rows and columns. The points
# need not lie on a regular grid: give their coordinates and each one stores its own x and y.
rows, columns = 4, 5
u, v = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
x = (u + 0.1 * np.sin(3 * v)) * 1e-3  # the columns bend with the row
y = v * 2e-3
z = 1e-6 * np.cos(4 * u + v)
irregular = x3pio.Surface.from_points(np.stack([x, y, z], axis=-1), metadata=metadata)
print("regular grid:", irregular.is_regular_grid, "stored per point:", irregular.layers[0].x.shape)

# The coordinates of a layer come out in the same shape that went in: (rows, columns, 3).
points = irregular.layers[0].points()
print(
    "points:",
    points.shape,
    "equal to the input:",
    bool(np.allclose(points, np.stack([x, y, z], -1))),
)

# A point cloud has no order and no neighbours: an (N, 3) array of x, y and z in metres.
cloud_points = np.random.default_rng(1).uniform(0.0, 1e-3, (5000, 3))
cloud = x3pio.PointCloud.from_points(cloud_points, metadata=metadata)
print(type(cloud).__name__, cloud.layer.points().shape)

# ``write_points`` writes the array to a file directly.
buffer = io.BytesIO()
x3pio.write_points(buffer, cloud_points, metadata=metadata)
print("read back:", x3pio.read(io.BytesIO(buffer.getvalue())).layers[0].points().shape)

# A point that is not measured is left out of a point cloud, since the list has no place for it.
# The storage type decides the precision of the file: 16 bits keep about 5 significant digits.
small = x3pio.PointCloud.from_points(cloud_points, data_type=x3pio.DataType.INT16)
stored = x3pio.PointCloud.loads(small.dumps())  # what the file holds
print("largest error with 16 bits:", float(np.abs(stored.layer.points() - cloud_points).max()))
