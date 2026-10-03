# x3pio

[![CI](https://github.com/aschet/x3pio/actions/workflows/ci.yml/badge.svg)](https://github.com/aschet/x3pio/actions/workflows/ci.yml)
[![Docs](https://github.com/aschet/x3pio/actions/workflows/docs.yml/badge.svg)](https://aschet.github.io/x3pio/)
[![PyPI](https://img.shields.io/pypi/v/x3pio.svg)](https://pypi.org/project/x3pio/)

<!-- docs-include-start -->

Read and write ISO 25178-72 x3p files (`.x3p`) in Python.

x3p is the format of ISO 25178-72:2017 and Amd 1:2020 for the storage and exchange of topography
and profile data. It stores profiles, surfaces on a regular or an irregular grid, in one or several
layers, and point clouds as 3-D coordinates in a right-handed coordinate system, together with the
metadata of the measurement and optional files of a vendor.

## Installation

```bash
pip install x3pio
```

## Usage

```python
import numpy as np
import x3pio

# The metadata describes the measurement. The standard asks for a creation date, which a new
# file sets to now if the record has none, and for the kind of probing system.
metadata = x3pio.Metadata(
    creator="Jane Doe",
    instrument=x3pio.Instrument("Acme Metrology", "Scanner 3000", "12345", "firmware 2.1"),
    probing_system=x3pio.ProbingSystem(x3pio.ProbingType.NON_CONTACTING, "confocal, 50x objective"),
    comment="Reference artefact",
)
heights = np.zeros((480, 640))  # in metres
x3pio.write("surface.x3p", heights, x_scale=1e-6, y_scale=1e-6, metadata=metadata)

x3p = x3pio.read("surface.x3p")
print(type(x3p).__name__, len(x3p.layers))  # Surface 1
print(x3p.layers[0].z.shape)                # (480, 640), the heights in metres
print(x3p.layers[0].points().shape)         # (480, 640, 3), x, y and z in metres
print(x3p.metadata.instrument.model)        # Scanner 3000
```

`x3pio.read` returns a `Profile`, a `Surface` or a `PointCloud`, according to the file. The data
is in the layers: `x3p.layers` holds them. A layer gives its heights `z`, its coordinates `x` and
`y` and its `points()`, which are in the global coordinate system. A point that was not measured is
`NaN`. A point cloud has exactly one layer, which is its `layer`, and `x3pio.write_points` writes
it.

The arrays of a layer belong to the file: change them in place and `save` the file again, or give
a file other layers with `with_layers`, which keeps its metadata.

Without `metadata`, a new file names x3pio as its software. `x3pio.info(path)` reads the
description of a file, its axes, shape, metadata and vendor files, without reading the data.

## Documentation

- [x3p Format](https://aschet.github.io/x3pio/format.html): what an x3p file contains.
- [Examples](https://aschet.github.io/x3pio/examples.html): a short script for each topic, in
  the `examples/` folder.

## Command Line

```bash
x3pio info surface.x3p
x3pio convert -s xml surface.x3p surface_xml.x3p
x3pio convert -t int16 surface.x3p surface_int16.x3p
x3pio convert -r ISO5436_2000 surface.x3p surface_2000.x3p
```

`convert` changes the storage form (`-s`, `xml` or `binary`), the z data type (`-t`) and the
revision (`-r`) independently and repairs deviations of the source. With `-f`/`--fix` it drops
what cannot be written instead of failing: metadata without a date or a probing type, and vendor
extensions when converting to ISO 25178-72:2017. The command line also runs as `python -m x3pio`.

## Development

```bash
python3 -m venv .venv  # Windows: python -m venv .venv
source .venv/bin/activate  # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -e . --group dev
pre-commit install
pytest
pre-commit run --all-files
pip-audit
```

## References

- ISO 25178-72:2017, [Geometrical product specifications (GPS) — Surface texture: Areal
  — Part 72: XML file format x3p](https://www.iso.org/standard/62310.html)
- ISO 25178-72:2017/Amd 1:2020, [Geometrical product specifications (GPS) — Surface
  texture: Areal — Part 72: XML file format x3p — Amendment 1](https://www.iso.org/standard/78432.html)
- [openGPS](https://sourceforge.net/projects/open-gps/)
