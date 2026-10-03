# Sample Files

Synthetic x3p files for each kind of data in `iso25178-72-fdam1/`. Every file exists with the
coordinates in a binary file (`_binary`) and as XML text (`_xml`), except `surface_extensions`,
which is binary only. `iso25178-72-2017/` has only `surface` and `surface_extensions`: the revision
differs in the `Revision` text, the calibration date and the vendor extensions, not in the data.

| Name | Data |
| --- | --- |
| `profile` | 256 points, float64 |
| `profile_layers` | 3 layers of 256 points, float32, with missing points in the second |
| `profile_int16` | 256 points with missing points, int16 |
| `profile_int32` | 256 points, int32 |
| `surface` | 200 × 150 points with a region of missing points, int16 |
| `surface_int32` | 120 × 90 points with a region of missing points, int32 |
| `surface_placed` | 120 × 90 points with a rotation of 10° around x and an offset, float64 |
| `surface_absolute_placed` | The same with absolute x and y |
| `surface_layers` | 3 layers of 160 × 120 points, float64 |
| `surface_irregular` | 96 × 72 points on an irregular grid, float64 |
| `surface_irregular_layers` | 3 layers of 80 × 60 points on an irregular grid, float64 |
| `surface_extensions` | 100 × 80 points, float32, with vendor extension files: two vendors, in the first revision one |
| `point_cloud` | 1000 points, float32 |
| `point_cloud_int16`, `point_cloud_int32`, `point_cloud_float64` | The same points with all coordinates in that type |

`generate.py` makes the files.
