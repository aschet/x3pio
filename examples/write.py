"""Write surfaces, profiles and point clouds: python examples/write.py [folder]."""

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import x3pio
from x3pio import CoordinateSystem, DataStorage, DataType, Placement, Revision

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
folder.mkdir(parents=True, exist_ok=True)

# The standard asks for a creation date, which a new file sets to now, and for the kind of probing.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Scanner 3000", "12345", "firmware 2.1"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.NON_CONTACTING, "confocal, 50x objective"),
    calibration_date=datetime(2026, 1, 15, 9, 30, tzinfo=UTC),
)

# Heights and increments are in metres; NaN marks a point that was not measured.
heights = 1e-6 * np.random.default_rng(0).normal(size=(480, 640))
heights[100:110, 200:210] = np.nan
x3pio.write(folder / "surface.x3p", heights, x_scale=1e-6, y_scale=1e-6, metadata=metadata)

# Repeated measurements are the layers of one file.
repeats = [heights, heights + 1e-7, heights + 2e-7]
x3pio.Surface.from_layers(repeats, x_scale=1e-6, y_scale=1e-6, metadata=metadata).save(
    folder / "surface_layers.x3p"
)

# A profile is a line of heights. One that does not lie on a line is given by its points.
position = np.linspace(0.0, 4.0, 200)
x3pio.write(folder / "profile.x3p", 1e-6 * np.sin(position), x_scale=20e-6, metadata=metadata)
turn = np.linspace(0.0, np.pi, 50)
path = np.column_stack([np.cos(turn), np.sin(turn), 1e-3 * np.cos(4 * turn)]) * 1e-3
x3pio.Profile.from_points(path, metadata=metadata).save(folder / "profile_curved.x3p")

# A surface keeps its rows and columns, but its points need not lie on a grid: each stores its own
# x and y.
u, v = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 4))
bent = np.stack([(u + 0.1 * np.sin(3 * v)) * 1e-3, v * 2e-3, 1e-6 * np.cos(4 * u + v)], axis=-1)
x3pio.Surface.from_points(bent, metadata=metadata).save(folder / "surface_irregular.x3p")

# A point cloud has no order and no neighbours: an (N, 3) array of x, y and z.
cloud = np.random.default_rng(1).uniform(0.0, 1e-3, (5000, 3))
x3pio.write_points(folder / "point_cloud.x3p", cloud, metadata=metadata)

# The storage type sets the precision and the size. The default z_scale, "auto", is the smallest
# that fits the heights into an integer type.
compact = x3pio.Surface.from_array(
    heights, x_scale=1e-6, y_scale=1e-6, z_type=DataType.INT16, metadata=metadata
)
for storage in DataStorage:
    compact.save(folder / f"surface_int16_{storage.name.lower()}.x3p", storage=storage)

# A placement says where the stored view coordinate system lies in the global one.
placement = Placement.translation(0.10, 0.02, 0.0) @ Placement.from_axis_angle([0, 0, 1], np.pi / 6)
x3pio.write(
    folder / "surface_placed.x3p",
    1e-6 * np.ones((3, 4)),
    x_scale=1e-3,
    y_scale=1e-3,
    placement=placement,
    coordinate_system=CoordinateSystem.VIEW,
    metadata=metadata,
)

# A vendor adds files to the container under a URI that it owns. Amd 1:2020, the default, gives
# each file an ID; ISO 25178-72:2017 has one ID per vendor.
extended = x3pio.Surface.from_array(np.zeros((4, 4)), x_scale=1e-6, y_scale=1e-6, metadata=metadata)
extended.extensions.add("http://www.acme.example", "notes/sample.txt", b"Measured on the bench.\n")
extended.save(folder / "surface_extensions.x3p")
x3pio.write(
    folder / "surface_iso5436.x3p",
    np.zeros((4, 4)),
    x_scale=1e-6,
    y_scale=1e-6,
    revision=Revision.ISO5436_2000,
    metadata=metadata,
)

print("written to", folder)
for file in sorted(folder.glob("*.x3p")):
    print(f"{file.name:28s}{file.stat().st_size:>10} bytes")
