"""
transformation Module

==========
This module contains methods to do transformation on different data points.


Capabilities:
----------------------
- Data structure class
-

Limitations:
-----------


Tests
-----

- The test functions are inside the test.py. One can also use them as example of how to use the functions.

"""
import IO
import numpy as np


class InferMorphologyTransformation:
    def __init__(self, src_points=None, dst_points=None, input_path=None, alignment_matrix=None, matrix=None,
                 points=None):
        self.input_path = input_path
        self.matrix = matrix
        self.src_points = src_points
        self.dst_points = dst_points
        self.points = points
        self.transformed_points = None

    def set_transformation_matrix_from_am_file(self):
        """
        This method can extract the amira transformation matrix written in am file.

        """
        input_path = self.input_path

        matrix = []
        vector = []
        row = []
        with open(input_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line.rfind("TransformationMatrix") > -1:
                    vector = IO.read_numbers_in_line(line)
            for i in range(4):
                for j in range(4):
                    k = j + i * 4
                    row.append(vector[k])
                matrix.append(row)
                row = []
        self.matrix = matrix

    def set_transformation_matrix_from_landmark_file(self, input_path=None):
        """
        This method can extract the amira transformation matrix written in am file.

        """
        if input_path is None:
            input_path = self.input_path
            assert input_path
        pair_landmark_points = IO.read_landmark_file(input_path)
        pair_landmark_points = [list(plp) for plp in pair_landmark_points]

        points_system_1 = pair_landmark_points[::2]
        points_system_2 = pair_landmark_points[1::2]

        self.set_transformation_matrix_explicitly(points_system_1, points_system_2)

    def set_transformation_matrix_explicitly(self, src_points=None, dst_points=None):
        """
        This function will calculate the affine transformation matrix from
        8 points (4 source points and 4 destination points)

        """
        if src_points is None:
            src_points = self.src_points
            assert src_points
        if src_points is None:
            dst_points = self.dst_points
            assert dst_points

        dst = dst_points
        src = src_points

        x = np.transpose(np.matrix([src[0], src[1], src[2], src[3]]))
        y = np.transpose(np.matrix([dst[0], dst[1], dst[2], dst[3]]))

        # add ones on the bottom of x and y
        x = np.matrix(np.vstack((x, [1.0, 1.0, 1.0, 1.0])))
        y = np.matrix(np.vstack((y, [1.0, 1.0, 1.0, 1.0])))
        # solve for A2

        matrix = y * x.I
        self.matrix = matrix

    def execute_transformation_matrix(self):
        """
        transforms the first 3 coordinates of the points.
        """
        points = self.points
        matrix = self.matrix

        transformed_points = []
        for point4D in points:
            point = point4D[:3]
            m_point = np.matrix(point)
            m_tr_point = m_point.T
            p = matrix * np.matrix(np.vstack((m_tr_point, 1.0)))
            p = np.array(p.T)
            p_listed = p.tolist()[0]
            transformed_points.append(p_listed[0:3] + point4D[3:])

        self.transformed_points = transformed_points


class ConvertPoints:
    """
    This class converts points from coordinate_2d to image_coordinate_2d adn vice versa
    inputs.
    x_res: The pixel size in micron for x axis
    y_res: The pixel size in micron for y axis
    z_res: The pixel size in micron for z axis

    """

    def __init__(self, x_res=0.092, y_res=0.092, z_res=1.0):
        self.x_res = x_res
        self.y_res = y_res
        self.z_res = z_res

    def coordinate_2d_to_image_coordinate_2d(self, coordinate_2d):
        """
        Converted points from coordinate_2d to image_coordinate_2d

        :param coordinate_2d: TYPE: coordinate_2d
        :return: image_coordinate_2d: TYPE: 2d list of floats, but converted in the unit of the image.

        """
        points = coordinate_2d
        scaling = [1.0 / self.x_res, 1.0 / self.y_res, 1.0 / self.z_res]
        return _scaling(points, scaling)

    def image_coordinate_2d_to_coordinate_2d(self, image_coordinate_2d):
        """
        Converted points from image_coordinate_2d to coordinate_2d

        :param image_coordinate_2d: TYPE: 2d list of floats, but converted in th
        :return: coordinate_2d: TYPE: 2d list of floats
        """
        points = image_coordinate_2d
        scaling = [self.x_res, self.y_res, self.z_res]
        return _scaling(points, scaling)


def _scaling(points, scaling):
    if points is None:
        points = points
        assert points
    out = []
    for lv, pp in enumerate(points):
        try:
            s = scaling[lv]
        except IndexError:
            s = 1
        out.append(pp * s)
    converted_points = out
    return converted_points


def get_distance(p1, p2):
    assert(len(p1) == len(p2))
    return np.sqrt(sum((pp1-pp2)**2 for pp1, pp2 in zip(p1, p2)))