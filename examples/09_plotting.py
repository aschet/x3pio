# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Plot the data of files with matplotlib."""

import io
from pathlib import Path

import x3pio

# The sample files of the repository, one of each kind; use the path of your own files instead.
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples" / "iso25178-72-fdam1"
surface = x3pio.Surface.open(SAMPLES / "surface_binary.x3p")
irregular = x3pio.Surface.open(SAMPLES / "surface_irregular_binary.x3p")
profile = x3pio.Profile.open(SAMPLES / "profile_layers_binary.x3p")
cloud = x3pio.PointCloud.open(SAMPLES / "point_cloud_binary.x3p")

# The heights are a NumPy array in metres, the distance between the points is ``spacing``, and
# ``extent`` gives the edges of the grid cells that image plots ask for.
layer = surface.layers[0]
print("spacing:", layer.spacing, "extent:", layer.extent)

# matplotlib is not a dependency of x3pio; the plots are made if it is installed. The values are
# in metres, so they are scaled for the labels. Points that were not measured are NaN and stay
# blank.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed, no plot")
else:
    figure = plt.figure(figsize=(14, 8), layout="constrained")

    # A surface as an image. ``y`` grows with the row, so the origin is in the lower left corner.
    left, right, bottom, top = (value * 1e3 for value in layer.extent)
    axes = figure.add_subplot(2, 3, 1)
    image = axes.imshow(layer.z * 1e6, origin="lower", extent=(left, right, bottom, top))
    figure.colorbar(image, ax=axes, label="z / um")
    axes.set(xlabel="x / mm", ylabel="y / mm", title="Surface: Height Map")

    # The same surface in 3-D, from the three slices of ``points()``; ``rstride`` and ``cstride``
    # are the options of matplotlib that draw every second cell.
    grid = layer.points() * [1e3, 1e3, 1e6]
    axes = figure.add_subplot(2, 3, 2, projection="3d")
    axes.plot_surface(
        grid[..., 0], grid[..., 1], grid[..., 2], rstride=2, cstride=2, cmap="viridis"
    )
    axes.set(xlabel="x / mm", ylabel="y / mm", title="Surface: 3-D View")
    axes.set_zlabel("z / um", labelpad=8)
    axes.set_box_aspect((4, 3, 1))  # the heights are much smaller than the lateral size

    # A row of the surface is a profile along x.
    axes = figure.add_subplot(2, 3, 3)
    axes.plot(layer.x_values * 1e3, layer.z[75] * 1e6)
    axes.set(xlabel="x / mm", ylabel="z / um", title="Surface: Row 75 as a Profile")

    # An irregular surface stores the position of every point. ``points()`` gives them in the
    # shape of the grid, which is what ``pcolormesh`` takes.
    points = irregular.layers[0].points() * [1e3, 1e3, 1e6]
    axes = figure.add_subplot(2, 3, 4)
    mesh = axes.pcolormesh(points[..., 0], points[..., 1], points[..., 2], shading="nearest")
    figure.colorbar(mesh, ax=axes, label="z / um")
    axes.set(xlabel="x / mm", ylabel="y / mm", title="Irregular Surface: Height Map")

    # A profile of several layers: the positions along the line are ``x_values``.
    axes = figure.add_subplot(2, 3, 5)
    for number, scan in enumerate(profile.layers, start=1):
        axes.plot(scan.x_values * 1e3, scan.z * 1e6, label=f"Layer {number}")
    axes.set(xlabel="x / mm", ylabel="z / um", title="Profile: Three Layers")
    axes.legend()

    # A point cloud as 3-D points, the columns of the (N, 3) array being x, y and z.
    cloud_points = cloud.layer.points() * [1e3, 1e3, 1e6]
    axes = figure.add_subplot(2, 3, 6, projection="3d")
    axes.scatter(*cloud_points.T, c=cloud_points[:, 2], s=4)
    axes.set(xlabel="x / mm", ylabel="y / mm", title="PointCloud: 3-D Points")
    axes.set_zlabel("z / um", labelpad=8)

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    print("plots written:", len(buffer.getvalue()) > 0)
