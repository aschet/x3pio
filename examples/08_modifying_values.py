# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Change the values of a file in place, or write other data under the same metadata."""

import numpy as np

import x3pio

rng = np.random.default_rng(1)
heights = rng.normal(0.0, 1e-7, (64, 64)) + np.linspace(0.0, 2e-6, 64)
surface = x3pio.Surface.from_array(heights, x_scale=1e-6, y_scale=1e-6)
layer = surface.layers[0]

# The arrays of a layer belong to the file. Level the heights and mask the outliers in place.
layer.z -= np.polyval(np.polyfit(layer.x_values, layer.z.mean(axis=0), 1), layer.x_values)
layer.z[np.abs(layer.z) > 2.5e-7] = np.nan
print(int((~layer.valid).sum()), "of", layer.z.size, "points are not measured")

# Assigning fills the array, so it needs the shape of the layer, or a single number.
try:
    layer.z = np.zeros(10)
except ValueError as error:
    print(error)

# A file that came from a ``with_`` method shares its arrays and is read-only; copy it to write.
shared = surface.with_metadata(comment="levelled")
editable = shared.copy()
editable.layers[0].z[0, 0] = 0.0
print(editable.metadata.comment)

# The result of an analysis often has another shape. Give it to ``with_layers`` to keep the
# metadata, the vendor extensions, the revision and the storage form of the file.
window = x3pio.SurfaceLayer.from_array(layer.z[16:48, 16:48], x_scale=1e-6, y_scale=1e-6)
cropped = shared.with_layers([window])
print(cropped.layers[0].z.shape, cropped.metadata.comment)

# Other points than a grid can have are layers too, made from the coordinates of the points.
points = layer.points()
points[..., 0] *= 1.5
stretched = shared.with_layers([x3pio.SurfaceLayer.from_points(points)])
print(stretched.is_regular_grid, stretched.metadata.comment)

# The layers of a profile or a surface go into a new file, also from several files and in any
# order. The new file has copies of them and the metadata that is given, or a new record.
stack = x3pio.Surface.from_layers([layer, editable.layers[0]], metadata=surface.metadata)
backwards = x3pio.Surface.from_layers(reversed(stack.layers))
print(len(stack.layers), backwards.layers[0] == stack.layers[1])
