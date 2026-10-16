<!--
SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>

SPDX-License-Identifier: MIT
-->

# Examples

The scripts are in the `examples/` folder.

## Writing

Surfaces, profiles, irregular surfaces and point clouds, with metadata, placement, vendor
extensions, different storage types and revisions.

```{literalinclude} ../examples/write.py
:language: python
```

## Reading

The layers, the metadata and the checksums of a file, from a path, a stream or bytes.

```{literalinclude} ../examples/read.py
:language: python
```

## Modifying

Changing the values, making files of parts of a file, converting it to another revision or storage
type, and moving it in the global coordinate system.

```{literalinclude} ../examples/modify.py
:language: python
```

## Plotting

```{literalinclude} ../examples/plot.py
:language: python
```
