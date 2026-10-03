.. SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
..
.. SPDX-License-Identifier: MIT

API Reference
=============

Reading and Writing
-------------------

.. automodule:: x3pio.application
   :no-members:

.. autofunction:: x3pio.read
.. autofunction:: x3pio.info
.. autoclass:: x3pio.FileDescription
.. autoclass:: x3pio.FileInfo
.. autofunction:: x3pio.write
.. autofunction:: x3pio.write_points

File Types
----------

.. automodule:: x3pio.model.files
   :no-members:

.. autoclass:: x3pio.X3pFile
   :members:
.. autoclass:: x3pio.Profile
   :members:
.. autoclass:: x3pio.Surface
   :members:
.. autoclass:: x3pio.PointCloud
   :members:
.. autoclass:: x3pio.ArrayOptions
.. autoclass:: x3pio.PointOptions
.. autoclass:: x3pio.MetadataChanges

Layers
------

.. automodule:: x3pio.model.layers
   :no-members:

.. autoclass:: x3pio.Layer
   :members:
.. autoclass:: x3pio.ProfileLayer
   :members:
   :inherited-members:
.. autoclass:: x3pio.SurfaceLayer
   :members:
   :inherited-members:
.. autoclass:: x3pio.PointLayer
   :members:
   :inherited-members:
.. autoclass:: x3pio.Point

Coordinates
-----------

.. automodule:: x3pio.model.geometry
   :no-members:

.. autoclass:: x3pio.CoordinateSystem
   :members:
.. autoclass:: x3pio.Placement
   :members:
.. autofunction:: x3pio.validate_rotation

Header and Metadata
-------------------

.. automodule:: x3pio.model.header
   :no-members:

.. autoclass:: x3pio.Header
.. autoclass:: x3pio.Axis
   :members:
.. autoclass:: x3pio.AxisType
.. autoclass:: x3pio.FeatureType
.. autoclass:: x3pio.Revision
   :members:
.. autoclass:: x3pio.Metadata
.. autoclass:: x3pio.Instrument
.. autoclass:: x3pio.ProbingSystem
.. autoclass:: x3pio.ProbingType
.. autofunction:: x3pio.validate_increment

Data Types and Storage
----------------------

.. automodule:: x3pio.model.datatypes
   :no-members:

.. autoclass:: x3pio.DataType
.. autoclass:: x3pio.DataTypeInfo
   :members:
.. py:data:: x3pio.DATA_TYPES

   The :class:`DataTypeInfo` of each :class:`DataType`.

.. autofunction:: x3pio.get_data_type
.. autofunction:: x3pio.suggest_scale
.. autofunction:: x3pio.encode_raw
.. autofunction:: x3pio.decode_raw
.. autoclass:: x3pio.DataStorage

Vendor Extensions
-----------------

.. automodule:: x3pio.model.extensions
   :no-members:

.. autoclass:: x3pio.VendorExtensions
   :members:

Files With Deviations
---------------------

.. automodule:: x3pio.codec
   :no-members:

Command Line
------------

.. automodule:: x3pio.cli
   :no-members:

Exceptions
----------

.. autoexception:: x3pio.X3pError
.. autoexception:: x3pio.X3pFormatError
.. autoexception:: x3pio.X3pChecksumError
