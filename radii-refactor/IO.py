"""
IO Module

==========
This module contains methods to read and write data with different formats.


File extensions
----------------------

1. .am
Capabilities:
- Reading data points in 3d and their corresponding radius
- Writing data points in 3d and their corresponding radius
Limitations:
- Not able to read other data than 3d positional points and their radius
- Only able to read and write in ascii format (In this module in the class amira_utils the are methods one can use to
 convert amira binary format to askii format, for more details look at the amira_utils class docstring)

2 .hoc
Capabilities:
- Reading data points in 3d and their corresponding radius
- Writing data points in 3d and their corresponding radius
Limitations:
- Not able to construct the tree structure from the hoc file.
- Not able to read the white matter data.

3 .hx
Capabilities:
- Read and Write data with the help of Amira.
Limitations:
- Need to connect to Amira.

Tests
-----

- The test functions are inside the test.py. One can also use them as example of how to use the functions.

"""