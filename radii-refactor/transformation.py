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
import re
import numpy as np


class Data:
    def __init__(self, coordinate_2d=None, coordinate=None, coordinate_with_radius=None, pixel_coordinate=None):
        self.coordinate_2d = coordinate_2d      # TYPE: 2d list of floats
        self.coordinate = coordinate        # TYPE: 3d list of floats
        self.coordinate_with_radius = coordinate_with_radius    # TYPE: 4d list of floats
        self.pixel_coordinate = pixel_coordinate        # TYPE: 2d list of integers


class Matrix:
    def __init__(self, src_morphology=None, dst_morphology=None, input_path=None, alignment_matrix=None, matrix=None,
                 points=None):
        self.input_path = input_path
        self.alignment_matrix = alignment_matrix
        self.matrix = matrix
        self.src_morphology = src_morphology
        self.dst_morphology = dst_morphology
        self.points = points
        self.transformed_points = None
        self.converted_points = None

    def read_from_file_for_alignment(self, input_path=None):
        """
        This method can extract the manual transformation matrix written in am file.

        """
        if input_path is None:
            input_path = self.input_path
            assert input_path

        matrix = []
        vector = []
        row = []
        with open(input_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                if line.rfind("TransformationMatrix") > -1:
                    matches = re.findall('-?\d+\.\d+|-?\d+', line)
                    vector = map(float, matches)
            for i in range(4):
                for j in range(4):
                    k = j + i * 4
                    row.append(vector[k])
                matrix.append(row)
                row = []
        self.alignment_matrix = matrix
        return self.alignment_matrix

    def find(self, src_morphology=None, dst_morphology=None):
        """
        This function will calculate the affine transformation matrix from
        8 points (4 source points and 4 destination points)

        """
        if src_morphology is None:
            src_morphology = self.src_morphology
            assert src_morphology
        if src_morphology is None:
            dst_morphology = self.dst_morphology
            assert dst_morphology

        dst = dst_morphology
        src = src_morphology

        x = np.transpose(np.matrix([src[0], src[1], src[2], src[3]]))
        y = np.transpose(np.matrix([dst[0], dst[1], dst[2], dst[3]]))

        # add ones on the bottom of x and y
        x = np.matrix(np.vstack((x, [1.0, 1.0, 1.0, 1.0])))
        y = np.matrix(np.vstack((y, [1.0, 1.0, 1.0, 1.0])))
        # solve for A2

        matrix = y * x.I
        self.matrix = matrix
        return self.matrix

    def apply(self, points=None, matrix=None):
        """
        transforms the first 3 coordinates of the points.
        """
        if points is None:
            points = self.points
            assert points

        if matrix is None:
            matrix = self.matrix
            assert matrix

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
        return self.transformed_points

    def convert_point(self, points=None, x_res=0.092, y_res=0.092, z_res=1.0):
        if points is None:
            points = self.points
            assert points

        out = []
        scaling = [x_res, y_res, z_res]
        for lv, pp in enumerate(points):
            try:
                s = scaling[lv]
            except IndexError:
                s = 1
            out.append(pp * s)
        self.converted_points = out
        return self.converted_points
