<!--
SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>

SPDX-License-Identifier: MIT
-->

# x3p Format

x3p is the format of ISO 25178-72:2017 and Amd 1:2020 for the storage and exchange of topography
and profile data. It stores profiles, surfaces on a regular or an irregular grid, in one or several
layers, and point clouds as 3-D coordinates in a right-handed coordinate system, together with the
metadata of the measurement and optional files of a vendor. The `Revision` element of a file names
the revision of the standard that it follows.

## Container

An x3p file is a zip archive with the extension `.x3p`. It contains:

| File | Content |
| --- | --- |
| `main.xml` | The description of the data, and the coordinates if they are stored as text. |
| `md5checksum.hex` | The MD5 checksum of `main.xml`. |
| `bindata/data.bin` | The coordinates in binary form. |
| `bindata/valid.bin` | One bit per point that says whether it is valid, for integer data. |
| files of vendors | Anything a vendor adds, under any other name. |

Only `main.xml` and `md5checksum.hex` are required, in the root of the archive. The names of the
binary files are a recommendation; `main.xml` stores their paths.

## main.xml

The document has four records:

| Record | Content |
| --- | --- |
| `Record1` | The header: `Revision`, `FeatureType`, the axes `CX`, `CY` and `CZ`, and an optional `Rotation`. |
| `Record2` | Optional: the metadata with date, creator, instrument, calibration date, probing system and comment. |
| `Record3` | The data: the size, and the coordinates as a `DataList` of text or a `DataLink` to the binary files with their checksums. |
| `Record4` | The name of the checksum file. |

`VendorSpecificID` elements register the files of vendors.

## Points, Matrices and Layers

A point has the three coordinates `x`, `y` and `z` in metres. The `FeatureType` says how the
points are arranged:

- `PCL`, a point cloud, is a plain list of points without any order. It is meant for data of
  unknown topology, such as the output of a coordinate measuring machine.
- `SUR`, a surface, is a matrix of points. The points keep their neighbourhood: the points next
  to a point in the matrix are next to it in space. The writer has to assure this.
- `PRF`, a profile, is a sequence of points. It is stored as a matrix with a single row. It need
  not lie on a straight line.

The position of a point in the matrix is given by the indices `u`, `v` and `w`. `SizeX`, `SizeY`
and `SizeZ` of `Record3` are the sizes of the matrix in `u`, `v` and `w`. The indices count
positions in the matrix, and `x`, `y` and `z` are positions in space. They are related only for
incremental `x` and `y` axes, where `u` determines `x` and `v` determines `y`.

A profile or a surface can have several layers, the `w` direction: a point is then also next to
the points at the same place in the adjacent layers. Most files have a single layer. A matrix of
4 × 4 points in one layer has `SizeX` 4, `SizeY` 4 and `SizeZ` 1, and a profile of 10 points in
two layers has `SizeX` 10, `SizeY` 1 and `SizeZ` 2. A point cloud has a `ListDimension` instead,
the number of points.

The points are stored with `u` as the fastest index, then `v`, then `w`.

## Axes

`Record1` describes each axis `x`, `y` and `z` with four elements.

`AxisType` is `I` for an incremental axis or `A` for an absolute axis:

- An incremental axis stores no coordinates. The coordinate is the increment `I` times the
  index of the point: `x = (u − 1) · I` in Amd 1:2020. ISO 25178-72:2017 equates `x` with `u` and
  does not say where the index starts. With incremental `x` and `y` the points lie on a regular
  grid and only `z` is stored. The standard recommends this whenever the spacing is regular, since
  it saves memory.
- An absolute axis stores a value for every point. With absolute `x` and `y` the grid is
  irregular, but the matrix keeps its neighbourhood, so a surface can follow any shape. A profile
  can follow any path.

The `z` axis is always absolute, and so are all three axes of a point cloud.

`DataType` is the type of the stored values: `I` for int16, `L` for int32, `F` for float32 and
`D` for float64. `Increment` is a positive length in metres, never zero. It is the step of an
incremental axis, and the factor that turns the stored values of an absolute axis into metres.
`Offset` is the distance of the stored data to the origin along the axis, in metres. Together
with the rotation it places the view coordinate system in the global one.

## Coordinate Systems

The stored coordinates are in the view coordinate system, in which the instrument or the software
described the points. The global coordinate system is the one the data set belongs to. Both are
three-dimensional, right-handed coordinate systems. The optional `Rotation`, a 3 × 3 matrix `R`,
and the offsets `O` of the axes place the view system in the global one. `R` must be a pure
rotation, without mirroring, scaling or shearing, so it keeps the handedness. With the stored
values scaled by the increments `I`:

```text
global = R · (I · stored) + O
```

The standard recommends an offset that centres the stored points on the origin.

## Missing Points

A point of a profile or a surface can be marked as not measured. A point cloud has no such
points. The marking depends on the storage:

- In text, the `Datum` of the point is empty.
- In float32 and float64, the `z` coordinate is NaN.
- In int16 and int32, no value can stand for missing, so `valid.bin` holds one bit per point in
  the order of the points, 1 for a valid point.

The `x` and `y` of a missing point are known from an incremental axis. For absolute axes the
standard recommends storing them if they are known, so that the point can be interpolated later.

## Text and Binary Storage

In text, `Record3` has a `DataList` with one `Datum` per point in the order of the matrix. A
`Datum` holds the coordinates of the absolute axes, separated by semicolons.

In binary form, `data.bin` holds the same values without separators, each in the `DataType` of
its axis with the least significant byte first, and `Record3` links the file and stores its MD5
checksum, as it does for `valid.bin`. The standard recommends the binary form for more than 10000
points.

## Vendor Extensions

A vendor can add files of its own to the container, such as the settings of the instrument. The
`VendorSpecificID` element registers them with a URI that the vendor owns, so that two vendors do
not use the same name. Software that does not know a `VendorSpecificID` ignores it and its files.
