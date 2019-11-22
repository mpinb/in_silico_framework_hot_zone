import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'configuration.conf')

"""Definitions of Data structures:
        1. coordinate_2d, TYPE: 2d list of floats
        2. image_coordinate_2d, TYPE: 2d list of floats, but converted in the unit of the image.
        3. coordinate, TYPE: 3d list of floats
        4. coordinate_with_radius, TYPE: 4d list of floats
        5. pixel_coordinate, TYPE: 2d list of integers
        6. src_points,
        7. ds_points,
"""