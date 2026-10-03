# Interoperability Samples

`opengps/` holds sample files of the [openGPS](https://sourceforge.net/p/open-gps/) x3p library
under their original names, licensed as the library is, under the LGPL-3.0-or-later.

| File | Content |
| --- | --- |
| `opengps/ISO5436-sample1.x3p` | A 4 × 4 surface as XML text, with the unit rotation written out. |
| `opengps/ISO5436-sample1_bin.x3p` | The same data as a `float64` binary file, with a validity file and `NaN` for the missing point. |
| `opengps/ISO5436-sample2.x3p` | A point cloud of 16 points with absolute x, y and z axes, as XML text. |
| `opengps/ISO5436-sample3.x3p` | A surface with an incremental x, an absolute `int16` y and an absolute z axis. |
| `opengps/ISO5436-sample4.x3p` | A surface with an `int16` z axis (increment 0.001, offset -0.5), as XML text. |
| `opengps/ISO5436-sample4_bin.x3p` | The same data as a binary file with a validity file. |
| `iso25178-72-fdam1.xml` | The `main.xml` example of ISO 25178-72:2017/Amd 1:2020, a 4 × 4 surface. |

The `.x3p` files follow ISO 25178-72:2017 and are valid against its schema. Samples 1 to 3 have no
`Increment` for the z axis, and samples 2 and 3 have an `Offset` there without one, which the schema
of Amd 1:2020 requires, so only the two `sample4` files are valid against it. Some files start with
a UTF-8 byte order mark.
