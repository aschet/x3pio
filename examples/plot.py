"""Plot the data of files with matplotlib: python examples/plot.py [figure.png]."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import x3pio

# The sample files of the repository, one of each kind; use the paths of your own files instead.
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples" / "iso25178-72-fdam1"
surface = x3pio.read(SAMPLES / "surface_binary.x3p")
irregular = x3pio.read(SAMPLES / "surface_irregular_binary.x3p")
profile = x3pio.read(SAMPLES / "profile_layers_binary.x3p")
cloud = x3pio.read(SAMPLES / "point_cloud_binary.x3p")

# Metres are scaled for the labels. Points that were not measured are NaN and stay blank.
figure = plt.figure(figsize=(14, 8), layout="constrained")

# ``extent`` gives the edges of the grid cells that image plots ask for. y grows with the row, so
# the origin is in the lower left corner.
layer = surface.layers[0]
left, right, bottom, top = (value * 1e3 for value in layer.extent)
axes = figure.add_subplot(2, 3, 1)
image = axes.imshow(layer.z * 1e6, origin="lower", extent=(left, right, bottom, top))
figure.colorbar(image, ax=axes, label="z / um")
axes.set(xlabel="x / mm", ylabel="y / mm", title="Surface: Height Map")

# The same surface in 3-D, from the points of the layer.
grid = layer.points() * [1e3, 1e3, 1e6]
axes = figure.add_subplot(2, 3, 2, projection="3d")
axes.plot_surface(grid[..., 0], grid[..., 1], grid[..., 2], rstride=2, cstride=2, cmap="viridis")
axes.set(xlabel="x / mm", ylabel="y / mm", title="Surface: 3-D View")
axes.set_zlabel("z / um", labelpad=8)
axes.set_box_aspect((4, 3, 1))  # the heights are much smaller than the lateral size

# A row of the surface is a profile along x.
axes = figure.add_subplot(2, 3, 3)
axes.plot(layer.x_values * 1e3, layer.z[75] * 1e6)
axes.set(xlabel="x / mm", ylabel="z / um", title="Surface: Row 75 as a Profile")

# An irregular surface stores the position of every point; ``points()`` has the shape of the grid,
# which ``pcolormesh`` takes.
points = irregular.layers[0].points() * [1e3, 1e3, 1e6]
axes = figure.add_subplot(2, 3, 4)
mesh = axes.pcolormesh(points[..., 0], points[..., 1], points[..., 2], shading="nearest")
figure.colorbar(mesh, ax=axes, label="z / um")
axes.set(xlabel="x / mm", ylabel="y / mm", title="Irregular Surface: Height Map")

# The layers of a profile share the positions along the line.
axes = figure.add_subplot(2, 3, 5)
for number, scan in enumerate(profile.layers, start=1):
    axes.plot(scan.x_values * 1e3, scan.z * 1e6, label=f"Layer {number}")
axes.set(xlabel="x / mm", ylabel="z / um", title="Profile: Three Layers")
axes.legend()

# A point cloud is an (N, 3) array of x, y and z.
cloud_points = cloud.layer.points() * [1e3, 1e3, 1e6]
axes = figure.add_subplot(2, 3, 6, projection="3d")
axes.scatter(*cloud_points.T, c=cloud_points[:, 2], s=4)
axes.set(xlabel="x / mm", ylabel="y / mm", title="PointCloud: 3-D Points")
axes.set_zlabel("z / um", labelpad=8)

if len(sys.argv) > 1:
    figure.savefig(sys.argv[1])
    print("figure written:", sys.argv[1])
else:
    plt.show()
