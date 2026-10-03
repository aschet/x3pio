# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT

"""Make the sample files of this folder: ``python tests/samples/generate.py``.

Every kind of x3p data is made in both revisions of the standard, once with the coordinates in a
binary file and once as XML text. The data is synthetic and does not depend on the version of
x3pio, so the files only change when this script does.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from x3pio import (
    CoordinateSystem,
    DataStorage,
    DataType,
    Instrument,
    Metadata,
    Placement,
    PointCloud,
    ProbingSystem,
    ProbingType,
    Profile,
    Revision,
    Surface,
    X3pFile,
)

FOLDER = Path(__file__).parent
#: The folder of each revision.
FOLDERS = {
    Revision.ISO5436_2000: FOLDER / "iso25178-72-2017",
    Revision.ISO25178_72_2017_DAM1: FOLDER / "iso25178-72-fdam1",
}
LAYERS = 3
STORAGES = {"binary": DataStorage.BINARY, "xml": DataStorage.XML}
#: The samples that exist only with binary coordinates: the XML text would add nothing.
BINARY_ONLY = {"surface_extensions"}
#: The first revision differs from the amendment only in the revision text, the calibration date and
#: the extensions, so it has only the samples that show that.
FIRST_REVISION = {"surface", "surface_extensions"}


def storages(revision: Revision, name: str) -> dict[str, DataStorage]:
    """Return the storage forms that a sample exists in, by file name suffix."""
    if name in BINARY_ONLY or revision is Revision.ISO5436_2000:
        return {"binary": DataStorage.BINARY}
    return STORAGES


_SETTINGS = (
    "http://www.example.org/x3pio",
    "settings.xml",
    b"<settings><objective>20</objective></settings>\n",
)
_LOG = ("http://www.example.org/x3pio", "logs/run.txt", b"Synthetic run, 20x objective.\n")
_NOTES = ("http://www.example.com/lab", "notes.txt", b"Measured on the synthetic bench.\n")

#: The vendor extensions of the sample with them: the vendor, the path below it and the content.
#: The first revision has a single vendor, the amendment any number.
EXTENSIONS = {
    Revision.ISO5436_2000: (_SETTINGS, _LOG),
    Revision.ISO25178_72_2017_DAM1: (_SETTINGS, _LOG, _NOTES),
}


def _metadata(comment: str) -> Metadata:
    return Metadata(
        date=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        creator="x3pio sample generator",
        instrument=Instrument("x3pio", "tests/samples/generate.py", "", "1"),
        probing_system=ProbingSystem(ProbingType.SOFTWARE, "synthetic data"),
        comment=comment,
    )


def _heights(x: np.ndarray, y: np.ndarray, layer: int) -> np.ndarray:
    """Return a smooth, bumpy height in metres, different for each layer."""
    shift = 0.4 * layer
    return 1e-6 * (
        np.cos(2.0 * np.pi * x / 24e-6 + shift) * np.cos(2.0 * np.pi * y / 16e-6 - shift)
        + 0.3 * np.sin(2.0 * np.pi * x / 9e-6)
    )


def build(revision: Revision) -> dict[str, X3pFile]:
    """Return the sample files of a revision by name, without the storage suffix."""
    files: dict[str, X3pFile] = {}

    t = np.arange(256) * 2e-6
    profile = 1e-6 * (np.sin(2.0 * np.pi * t / 80e-6) + 0.2 * np.sin(2.0 * np.pi * t / 17e-6))
    files["profile"] = Profile.from_array(
        profile, x_scale=2e-6, revision=revision, metadata=_metadata("A profile of one layer.")
    )
    layers = [profile * (1 + 0.2 * layer) for layer in range(LAYERS)]
    layers[1][100:103] = np.nan
    files["profile_layers"] = Profile.from_layers(
        layers,
        x_scale=2e-6,
        z_type=DataType.FLOAT32,
        revision=revision,
        metadata=_metadata("A profile of three layers, with missing points in the second."),
    )
    for data_type in (DataType.INT16, DataType.INT32):
        values = profile.copy()
        if data_type is DataType.INT16:
            values[60:64] = np.nan
        files[f"profile_{data_type.name.lower()}"] = Profile.from_array(
            values,
            x_scale=2e-6,
            z_type=data_type,
            revision=revision,
            metadata=_metadata(f"A profile with heights in {data_type.name.lower()}."),
        )

    y, x = np.mgrid[0:150, 0:200] * 1e-6
    surface = _heights(x, y, 0)
    surface[40:52, 70:85] = np.nan
    files["surface"] = Surface.from_array(
        surface,
        x_scale=1e-6,
        y_scale=1e-6,
        z_type=DataType.INT16,
        revision=revision,
        metadata=_metadata("A surface of one layer with missing points, heights in int16."),
    )
    y, x = np.mgrid[0:90, 0:120] * 1e-6
    surface = _heights(x, y, 0)
    surface[30:36, 40:50] = np.nan
    files["surface_int32"] = Surface.from_array(
        surface,
        x_scale=1e-6,
        y_scale=1e-6,
        z_type=DataType.INT32,
        revision=revision,
        metadata=_metadata("A surface with missing points, heights in int32."),
    )
    # A rotation of 10 degrees around x and an offset, for incremental and for absolute axes.
    placement = Placement.translation(1e-6, 5e-3, 1.0) @ Placement.from_axis_angle(
        [1.0, 0.0, 0.0], np.radians(10.0)
    )
    y, x = np.mgrid[0:90, 0:120] * 1e-6
    heights = _heights(x, y, 0)
    files["surface_placed"] = Surface.from_array(
        heights,
        x_scale=1e-6,
        y_scale=1e-6,
        placement=placement,
        coordinate_system=CoordinateSystem.VIEW,
        revision=revision,
        metadata=_metadata("A surface with a rotation and an offset."),
    )
    files["surface_absolute_placed"] = Surface.from_points(
        np.stack([x, y, heights], axis=-1),
        placement=placement,
        coordinate_system=CoordinateSystem.VIEW,
        revision=revision,
        metadata=_metadata("A surface with absolute x and y, a rotation and an offset."),
    )
    y, x = np.mgrid[0:120, 0:160] * 1e-6
    files["surface_layers"] = Surface.from_layers(
        [_heights(x, y, layer) for layer in range(LAYERS)],
        x_scale=1e-6,
        y_scale=1e-6,
        revision=revision,
        metadata=_metadata("A surface of three layers."),
    )

    v, u = np.mgrid[0:72, 0:96]
    xs = (u + 0.4 * np.sin(v / 3.0)) * 2e-6
    ys = (v + 0.3 * np.cos(u / 4.0)) * 2e-6
    grids = [np.stack([xs, ys, _heights(xs, ys, layer)], axis=-1) for layer in range(LAYERS)]
    files["surface_irregular"] = Surface.from_points(
        grids[0],
        revision=revision,
        metadata=_metadata("A surface of one layer on an irregular grid."),
    )
    v, u = np.mgrid[0:60, 0:80]
    xs = (u + 0.4 * np.sin(v / 3.0)) * 2e-6
    ys = (v + 0.3 * np.cos(u / 4.0)) * 2e-6
    grids = [np.stack([xs, ys, _heights(xs, ys, layer)], axis=-1) for layer in range(LAYERS)]
    files["surface_irregular_layers"] = Surface.from_points(
        np.stack(grids),
        revision=revision,
        metadata=_metadata("A surface of three layers on an irregular grid."),
    )

    y, x = np.mgrid[0:80, 0:100] * 1e-6
    extended = Surface.from_array(
        _heights(x, y, 0),
        x_scale=1e-6,
        y_scale=1e-6,
        z_type=DataType.FLOAT32,
        revision=revision,
        metadata=_metadata(
            f"A surface with the files of {len({e[0] for e in EXTENSIONS[revision]})} vendor(s)."
        ),
    )
    for vendor, path, content in EXTENSIONS[revision]:
        extended.extensions.add(vendor, path, content)
    files["surface_extensions"] = extended

    rng = np.random.default_rng(1)
    xy = rng.uniform(0.0, 100e-6, (1000, 2))
    cloud = np.column_stack([xy, _heights(xy[:, 0], xy[:, 1], 0)])
    files["point_cloud"] = PointCloud.from_points(
        cloud,
        data_type=DataType.FLOAT32,
        revision=revision,
        metadata=_metadata("A point cloud of 1000 points."),
    )
    for data_type in (DataType.INT16, DataType.INT32, DataType.FLOAT64):
        files[f"point_cloud_{data_type.name.lower()}"] = PointCloud.from_points(
            cloud,
            data_type=data_type,
            revision=revision,
            metadata=_metadata(f"A point cloud with all coordinates in {data_type.name.lower()}."),
        )
    if revision is Revision.ISO5436_2000:
        files = {name: file for name, file in files.items() if name in FIRST_REVISION}
    return files


def main() -> None:
    """Write the files into the folders of the revisions."""
    for revision, folder in FOLDERS.items():
        folder.mkdir(exist_ok=True)
        for name, file in build(revision).items():
            for suffix, storage in storages(revision, name).items():
                file.save(folder / f"{name}_{suffix}.x3p", storage=storage)


if __name__ == "__main__":
    main()
