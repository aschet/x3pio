# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Write and read data with a placement in the global coordinate system."""

import numpy as np

import x3pio
from x3pio import CoordinateSystem, Placement

# A file stores its coordinates in the view coordinate system. A placement, a rotation and an
# offset, says where that system lies in the global coordinate system.
placement = Placement.translation(0.10, 0.02, 0.0) @ Placement.from_axis_angle([0, 0, 1], np.pi / 6)

# The placement is only numbers; the comment tells what the global system is.
metadata = x3pio.Metadata(
    instrument=x3pio.Instrument("Acme Metrology", "Scanner 3000", "12345", "2.1"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.NON_CONTACTING, "confocal"),
    comment="Global system: the table of the stage, in metres",
)

surface = x3pio.Surface.from_array(
    1e-6 * np.ones((3, 4)),
    x_scale=1e-3,
    y_scale=1e-3,
    placement=placement,
    coordinate_system=CoordinateSystem.VIEW,
    metadata=metadata,
)

# The stored coordinates are not changed; the coordinate system says which system the points are in.
view = surface.layers[0].points(coordinate_system=CoordinateSystem.VIEW).reshape(-1, 3)
print("view  :", view[1])
print(
    "global:", surface.layers[0].points(coordinate_system=CoordinateSystem.GLOBAL).reshape(-1, 3)[1]
)
print("the same, from the placement:", placement.apply(view)[1])

# Points in the global system are given with ``CoordinateSystem.GLOBAL``. They are stored in the
# view system, and ``points()`` of the layer returns them again.
measured = np.random.default_rng(2).uniform(0.0, 1e-3, (100, 3)) + np.array([0.1, 0.02, 0.0])
cloud = x3pio.PointCloud.from_points(
    measured, placement=placement, coordinate_system=CoordinateSystem.GLOBAL, metadata=metadata
)
print("the global points are kept:", bool(np.allclose(cloud.layer.points(), measured)))

# Moving a file. ``with_placement`` puts the stored data somewhere else; ``transformed`` moves
# the points relative to where they are now.
lifted = surface.transformed(Placement.translation(0.0, 0.0, 1e-3))
height = surface.layers[0].points()[0, 0, 2]
print("lifted by 1 mm:", float(lifted.layers[0].points()[0, 0, 2] - height))

# ``centered`` moves the stored coordinates of the absolute axes to the origin and the offset
# by as much, as the standard recommends, so the global points stay.
off_centre = x3pio.PointCloud.from_points(measured + 100.0)
centered = off_centre.centered()
print("stored x now between", float(centered.layer.x.min()), "and", float(centered.layer.x.max()))
print(
    "global points unchanged:",
    bool(np.allclose(centered.layer.points(), off_centre.layer.points())),
)

# ``globalized`` makes a copy whose stored coordinates are the global ones and whose placement is
# the identity.
baked = surface.globalized()
print("after globalized(), the placement is the identity:", baked.placement.is_identity)
