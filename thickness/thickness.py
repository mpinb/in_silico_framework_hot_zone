"""
Thickness Module

==========
This module contains methods to do extract thicknesses from image.


Capabilities:
----------------------
-
-

Limitations:
-----------


Tests
-----

- The test functions are inside the test.py. One can also use them as example of how to use the functions.

"""
import sys

import numpy as np
import warnings
import SimpleITK as sitk
import transformation as tr
from utils import get_size_of_object
import utils as u


class ThicknessExtractor:
    def __init__(self, points, image_file, xy_resolution=0.092, z_resolution=0.5, ray_length_front_to_back_in_micron=20,
                 number_of_rays=36, threshold_percentage=0.5, max_seed_correction_radius_in_micron=10):
        """ This is the main method for extracting Thickness
        - Inputs:
            1. points: must be in the type transformation.Data.coordinate_2d, so they are a list of
            2d values in micrometer
            2. Image is the path to tif file with a pixel size of xy_resolution.
            3. xy_resolution: pixelSize in micrometers as acquired by the microscope in one optical section.
            4. z_resolution: z distance in micrometers between optical sections
            5. ray_length_front_to_back_in_micron: maximum distance from the seed point considered in micrometer.

        """
        self.original_points = points
        self.convert_points = tr.ConvertPoints(xy_resolution, xy_resolution, z_resolution)
        points_in_image_coordinates_2d = self.convert_points.coordinate_2d_to_image_coordinate_2d(self.original_points)
        self.points = points_in_image_coordinates_2d  # image_coordinate_2d, TYPE: 2d list of floats,
        # but converted in the unit of the image.
        self.seed_corrected_points = []
        # TYPE: Must be transformation.Data.image_coordinate_2d
        self.image_file = image_file
        self.image = _read_image(image_file)
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution
        # ray_length_front_to_back_in_micron, is the size of total ray in micron.
        self.ray_length_front_to_back_in_micron = ray_length_front_to_back_in_micron
        self.number_of_rays = number_of_rays
        self._ray_length_per_direction_in_micron = ray_length_front_to_back_in_micron / 2.0
        # __ray_length_per_direction_of_image_coordinates: by dividing _ray_length_per_direction_in_micron with the
        # xy_resolution we go to the image coordinate system.
        # So the numbers in this coordinate are indication of pixels.
        self._ray_length_per_direction_of_image_coordinates = self._ray_length_per_direction_in_micron / xy_resolution
        self.threshold_percentage = threshold_percentage
        self._max_seed_correction_radius_in_image_coordinates = max_seed_correction_radius_in_micron / xy_resolution
        self._max_seed_correction_radius_in_image_coordinates_in_pixel = \
            int(self._max_seed_correction_radius_in_image_coordinates)

        self.padded_image = _pad_image(self.image, self._max_seed_correction_radius_in_image_coordinates_in_pixel)
        self.contour_list = []
        self.all_data = {}

        self.get_all_data_by_points()

    def get_all_data_by_points(self):
        """
        This is the main method of the class.
        To extract the thicknesses of points from the image, after initiating the class, this method need to be called.

        """

        points = self.points
        all_data = {}
        for idx, point in enumerate(points):
            #         print str(idx) + " points from " + str(len(points)) + " are completed."
            #        sys.stdout.write("\033[F")
            data = self.get_all_data_by_point(point)
            all_data[idx] = data

        overlaps = self.get_overlpas()
        print "size of object in MB all_data: " + str(get_size_of_object(all_data) / (1024. * 1024.))

        self.all_data = all_data
        self._tidy_up()

    def get_all_data_by_point(self, point):
        """
        Computes coordinates of rays and intensity profiles for one point.

        :param point: The TYPE Must be transformation.coordinate_2d.
        so the point is in Micrometer unit.
        :return: A dictionary of points as keys and all_data as the value.
        all_data itself is a dictionary of:
        1. back_profile, 2. front_profile, 3. thicknesses_list,
        4. min_thickness, 5. back_contour_index, 6. front_contour_index,
        7. contour_list, 8. rays_intensity_profile, 9. rays_indices, 10. selected_profile_index

        """

        all_data = {"converted_point_by_image_coordinate": point}
        original_point = self.convert_points.image_coordinate_2d_to_coordinate_2d([point])
        all_data["original_points"] = original_point
        if self._max_seed_correction_radius_in_image_coordinates:
            point = self._correct_seed(point)
        all_data["seed_corrected_point"] = point
        thicknesses_list = []
        min_thickness = np.Inf
        contour_list = []
        rays_intensity_profile = []
        rays_indices = []

        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            front_ray_indices = self.get_ray_points_indices(point, phi, front=True)
            back_ray_indices = self.get_ray_points_indices(point, phi, front=False)

            ray_indices = _construct_ray_from_half_rays(front_ray_indices, back_ray_indices, point)
            rays_indices.append(ray_indices)

            ray_intensity_profile = self.get_intensity_profile_from_ray_indices(ray_indices)
            rays_intensity_profile.append(ray_intensity_profile)

        # all_data["rays_indices"] = rays_indices
        # all_data["rays_intensity_profile"] = rays_intensity_profile

        for i, ray_indices in enumerate(rays_indices):

            ray_length = len(ray_indices)
            half_ray_length = (ray_length - 1) / 2

            back_contour_index = self.get_contour_index(point, list(reversed(ray_indices[0:half_ray_length + 1])))
            front_contour_index = self.get_contour_index(point, ray_indices[half_ray_length:ray_length])
            all_data["back_contour_index"] = back_contour_index
            all_data["front_contour_index"] = front_contour_index
            if len(back_contour_index) == 2 and len(front_contour_index) == 2:
                thickness = tr.get_distance(back_contour_index, front_contour_index)
            else:
                thickness = 0
            # assert (len(back_contour_index) == 2)
            # assert (len(front_contour_index) == 2)
            contour_list.append([back_contour_index, front_contour_index])

            thicknesses_list.append(thickness)

            if thickness < min_thickness:
                min_thickness = thickness
                all_data["min_thickness"] = min_thickness
                all_data["selected_ray_index"] = i

        all_data["contour_list"] = contour_list
        all_data["thicknesses_list"] = thicknesses_list
        return all_data

    def get_intensity_profile_from_ray_indices(self, ray_indices):
        image = self.image
        profile_values = []
        profile_indices_length = len(ray_indices)
        for i in range(profile_indices_length):
            try:
                pixel = map(lambda x: (int(x)), ray_indices[i])
                intensity_value = image.GetPixel(pixel)
            except RuntimeError as error:
                warnings.warn(error)
                intensity_value = 0
            profile_values.append(intensity_value)
        return profile_values

    def get_contour_index(self, point, ray_indices):

        image = self.image
        try:
            point_value = image.GetPixel([int(point[0]), int(point[1])])
        except RuntimeError as error:
            warnings.warn("Point outside the image! Assuming intensity = 0")
            point_value = 0

        # pointHalfValue = point_value/2.0
        point_threshold_value = point_value * self.threshold_percentage

        profile_indices_length = len(ray_indices)
        contour_indices = []
        for i in range(profile_indices_length - 1):

            try:
                pixel_1_value = image.GetPixel(ray_indices[i])
            except IndexError as error:
                warnings.warn(error)
                pixel_1_value = 0.0

            try:
                pixel_2_value = image.GetPixel(ray_indices[i + 1])
            except IndexError as error:
                warnings.warn(error)
                pixel_2_value = 0.0

            if pixel_1_value >= point_threshold_value >= pixel_2_value:
                contour_indices = ray_indices[i]
                break

        return contour_indices

    def get_ray_points_indices(self, point, phi, front):
        """
        This method will get a point in image coordinate system, and will return the ray indices from that point
         by the value of phi.
        :param point:
        :param phi:
        :param front:
        :return: ray indices. They are in pixel in image coordinate system.
        """
        image = self.image
        ray_length = self._ray_length_per_direction_of_image_coordinates

        ray_points_indices = []

        image_width = image.GetWidth()
        image_height = image.GetHeight()

        x_i = point[0]
        y_i = point[1]

        x_f = x_i
        y_f = y_i

        for index in range(int(ray_length)):

            if front:
                x_f = x_f + 1
            else:
                x_f = x_f - 1

            x_f = x_f - x_i
            y_f = y_f - y_i

            x_new = int(y_f * np.sin(phi) + x_f * np.cos(phi))
            y_new = int(y_f * np.cos(phi) - x_f * np.sin(phi))

            x_new = x_new + x_i
            y_new = y_new + y_i

            x_f = x_f + x_i
            y_f = y_f + y_i

            if x_new <= 1 or y_new <= 1 or x_new >= image_width or y_new >= image_height:
                break
            else:
                ray_points_indices.append([int(x_new), int(y_new)])

        return ray_points_indices

    def _correct_seed(self, point):
        radius = self._max_seed_correction_radius_in_image_coordinates_in_pixel
        center = point

        cropped_image = _crop_image(self.padded_image, center, radius, circle=True)
        indices_of_max_value = np.argwhere(cropped_image == np.amax(cropped_image))

        del cropped_image
        corrected_point = [indices_of_max_value[0][0] + point[0], indices_of_max_value[0][1] + point[1]]

        return corrected_point

    def _tidy_up(self):
        del self.image
        del self.padded_image
        del self.points
        del self.convert_points
        del self.contour_list

    def get_overlaps(self):
        self.all_data.keys()
        cubes = [u.get_neighbours_of_point(point, self.points, width=2) for point in self.points]
        for cube in cubes:
            for main_point in cube:
                for neighbour_point in cube:
                    if main_point is neighbour_point:
                        continue
                    
                point_1 = cube[idx]
                point_2 = cube[idx + 1]
        overlaps = u.find_overlaps(self.slice_thicknesses_object)


def _circle_filter(x, y, r):
    if x ** 2 + y ** 2 <= r ** 2:
        return 1
    else:
        return 0


def _pad_image(image, radius):
    image_array = sitk.GetArrayFromImage(image)
    return np.pad(image_array, radius, 'constant', constant_values=0)


def _crop_image(image_array, center, radius, circle=False):
    c1, c2 = int(center[0]), int(center[1])
    return_ = image_array[c1:c1 + 2 * radius + 1, c2:c2 + 2 * radius + 1]
    # return_ = b_pad[c1:c1 + 2 * radius + 1, c2:c2 + 2 * radius + 1]

    if circle:
        return_ = [[value * _circle_filter(row_lv - radius, col_lv - radius, radius)
                    for col_lv, value in enumerate(row)]
                   for row_lv, row in enumerate(return_)]
        return_ = np.array(return_)
    return return_


def _construct_ray_from_half_rays(front_ray_indices, back_ray_indices, point):
    """puts together two half rays and center point and returns full ray.

    front_ray_indices: List of lists of two integers reflecting the x-y-pixel position for the front_ray
    back_ray_indices: as above, for the back_ray
    point: center point of the ray (list of two integers)

    """
    center_point_index = [int(point[0]), int(point[1])]
    ray = list(reversed(back_ray_indices)) + [center_point_index] + front_ray_indices
    return ray


def _read_image(image_file):
    """Reading image file """
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetFileName(image_file)
    image = image_file_reader.Execute()
    return image
