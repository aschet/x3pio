"""Change a file, make files of parts of it, and convert it: python examples/modify.py."""

import numpy as np

import x3pio
from x3pio import DataType, Placement, Revision

heights = np.random.default_rng(1).normal(0.0, 1e-7, (64, 64)) + np.linspace(0.0, 2e-6, 64)
surface = x3pio.Surface.from_array(heights, x_scale=1e-6, y_scale=1e-6)
layer = surface.layers[0]

# The arrays of a layer belong to the file, so changing them changes it. Assigning fills the
# array: it needs the shape of the layer, or a single number.
layer.z -= np.polyval(np.polyfit(layer.x_values, layer.z.mean(axis=0), 1), layer.x_values)
layer.z[np.abs(layer.z) > 2.5e-7] = np.nan
print(int((~layer.valid).sum()), "of", layer.z.size, "points are not measured")

# The ``with_`` methods make a copy that shares the arrays read-only; ``copy`` makes one to write.
levelled = surface.with_metadata(comment="levelled")
editable = levelled.copy()
editable.layers[0].z[0, 0] = 0.0

# The result of an analysis often has another shape. ``with_layers`` keeps the metadata, the
# vendor extensions, the revision and the storage form of the file.
window = x3pio.SurfaceLayer.from_array(layer.z[16:48, 16:48], x_scale=1e-6, y_scale=1e-6)
cropped = levelled.with_layers([window])
print("crop:", cropped.layers[0].z.shape, cropped.metadata.comment)

# The layers of several files, in any order, make a new file.
stack = x3pio.Surface.from_layers([layer, editable.layers[0]], metadata=surface.metadata)
print("stack:", len(stack.layers), "layers")

# A file in another revision or with another storage type of the heights. The vendor extensions
# are converted to Amd 1:2020 but not to ISO 25178-72:2017; ``extensions_fit`` tells which.
surface.extensions.add("http://www.acme.example", "notes/sample.txt", b"Measured on the bench.\n")
print("extensions fit ISO 25178-72:2017:", surface.extensions_fit(Revision.ISO5436_2000))
older = surface.with_revision(Revision.ISO5436_2000, drop_extensions=True)
compact = older.with_z_type(DataType.INT16)
print(older.revision.name, compact.header.z.data_type.name, compact.header.z.increment)

# A file stores its coordinates in the view coordinate system, and a placement says where that lies
# in the global one. ``transformed`` moves the points relative to where they are now.
placement = Placement.translation(0.10, 0.02, 0.0) @ Placement.from_axis_angle([0, 0, 1], np.pi / 6)
placed = surface.with_placement(placement)
lifted = placed.transformed(Placement.translation(0.0, 0.0, 1e-3))
rise = lifted.layers[0].points()[0, 0, 2] - placed.layers[0].points()[0, 0, 2]
print(f"lifted by {rise * 1e3:.3g} mm")

# ``globalized`` stores the global coordinates and, once there is a rotation, an identity placement.
# ``centered`` moves the stored coordinates of absolute axes to the origin and the offset by as
# much; the points stay.
print("identity after globalized:", placed.globalized().placement.is_identity)
cloud = x3pio.PointCloud.from_points(np.random.default_rng(2).uniform(0.0, 1e-3, (100, 3)) + 100.0)
print(
    "points unchanged by centered:",
    bool(np.allclose(cloud.centered().layer.points(), cloud.layer.points())),
)
