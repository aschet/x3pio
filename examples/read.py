"""Read a file and look at its layers, metadata and checksums: python examples/read.py."""

import io
import zipfile
from pathlib import Path

import x3pio

# The sample files of the repository, one of each kind; use the paths of your own files instead.
SAMPLES = Path(__file__).resolve().parent.parent / "tests" / "samples" / "iso25178-72-fdam1"

# ``read`` takes a path or an open file and returns what the file holds: a Surface, a Profile or
# a PointCloud. They share the layers, the placement, the metadata and the revision.
for name in ("surface_binary", "profile_layers_binary", "point_cloud_binary"):
    x3p = x3pio.read(SAMPLES / f"{name}.x3p")
    shapes = [layer.points().shape for layer in x3p.layers]
    print(f"{type(x3p).__name__:10}", x3p.revision.name, shapes)

# Heights and coordinates are in metres, and NaN marks a point that was not measured. The axes of a
# regular grid follow from the increments.
surface = x3pio.read(SAMPLES / "surface_binary.x3p")
layer = surface.layers[0]
print("measured points:", int(layer.valid.sum()), "of", layer.z.size)
print("x runs to", layer.x_values[-1], "m, y to", layer.y_values[-1], "m")

# The metadata is optional, and a field may be empty.
if surface.metadata is not None:
    print(surface.metadata.instrument.manufacturer, surface.metadata.creator)

# ``info`` reads the description of a file and leaves its data alone.
summary = x3pio.info(SAMPLES / "profile_layers_binary.x3p")
print(summary.feature_type.name, summary.layer_count, "layers of", summary.shape)

# Streams and bytes work as well as paths.
blob = (SAMPLES / "surface_binary.x3p").read_bytes()
from_stream = x3pio.read(io.BytesIO(blob))
from_bytes = x3pio.X3pFile.loads(blob)
print("same data from a stream and from bytes:", from_stream == from_bytes == surface)

# A file that was changed after it was written fails its MD5 checksum, unless verify is False.
source = zipfile.ZipFile(io.BytesIO(blob))
changed = io.BytesIO()
with zipfile.ZipFile(changed, "w") as target:
    for member in source.namelist():
        content = source.read(member)
        target.writestr(member, content + b"\n" if member == "main.xml" else content)
try:
    x3pio.read(io.BytesIO(changed.getvalue()))
except x3pio.X3pChecksumError as error:
    print("changed after it was written:", error)
print("without the check:", x3pio.read(io.BytesIO(changed.getvalue()), verify=False).revision.name)

# All errors of x3pio are X3pError, and a bad file is a ValueError.
try:
    x3pio.read(io.BytesIO(b"not an x3p file at all"))
except x3pio.X3pError as error:
    print("not read:", error)
