"""
radii Module

==========
This module contains methods to do extract radii from image.


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

import numpy as np
import warnings
import SimpleITK as sitk
import transformation as tr



def construct_ray_from_half_rays(front_ray_indices, back_ray_indices, point):
    center_point_index = [int(point[0]), int(point[1])]
    ray = list(reversed(back_ray_indices)) + [center_point_index] + front_ray_indices
    return ray


def _read_image(image_file):
    """Reading image file """
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetFileName(image_file)
    image = image_file_reader.Execute()
    return image


def get_index_of_maximum(param):
    pass


class Radius_extractor:
    def __init__(self, points, image_file, xy_resolution=0.092, z_resolution=0.5, ray_length_front_to_back_in_micron=20,
                 number_of_rays=36, threshold_percentage=0.5, max_seed_correction_radius_in_micron=20):
        """ This is the main method for extracting radii
        - Inputs:
            1. points: must be in the type transformation.Data.coordinate_2d, so they are a list of
            2d values in micrometer
            2. image is a tif file with the relative resolution of xy_resolution to the points
             which are in micrometer

            3. xy_resolution: pixelSize in micrometers as acquired by the microscope in one optical section.
            4. z_resolution: z distance in micrometers between optical sections
            5. ray_length_front_to_back_in_micron: maximum distance from the seed point considered in micrometer.

        """
        points_in_image_coordinates = tr.convert_point(points, xy_resolution, xy_resolution, z_resolution)
        self.points = tr.Data(image_coordinate_2d=points_in_image_coordinates).image_coordinate_2d
        # TYPE: Must be transformation.Data.image_coordinate_2d
        self.points_with_intensity = []
        self.recorded_seed_corrected_points = []
        self.seed_corrected_points_with_intensity = []
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
        self.contour_list = []
        self.profile_data = {}
        self.all_data = {}

    def get_intensity_profiles_by_point(self, point):
        """
        This method is useful when one needs the ray and rays intensity profiles of just one point in the image
        provided in at init.

        Inputs:
        _____
        point: The TYPE Must be transformation.coordinate_2d.
        """
        point = tr.convert_point(point)
        if self._max_seed_correction_radius_in_image_coordinates:
            point = self._correct_seed(point)
        rays_profiles = []
        rays = []
        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            front_coordinates = self.get_ray_points_indices(point, phi, front=True)
            back_coordinates = self.get_ray_points_indices(point, phi, front=False)

            ray = construct_ray_from_half_rays(front_coordinates, back_coordinates, point)
            rays.append(ray)

            ray_profile = self.get_intensity_profile_from_ray_indices(ray)
            rays_profiles.append(ray_profile)

        return rays, rays_profiles

    def get_all_data_by_points(self):
        """
        This is the main method of the class.
        To extract the radius of points from the image, after initiating the class, this method need to be called.

        """

        points = self.points
        all_data = {}

        for point in points:
            data = self.get_all_data_by_point(point)
            all_data[point] = data

        self.all_data = all_data
        return self.all_data

    def get_all_data_by_point(self, point):
        """
        :param point: The TYPE Must be transformation.coordinate_2d.
        so the point is in Micrometer unit.
        :return: A dictionary of points as keys and all_data as the value.
        all_data itself is a dictionary of:
        1. back_profile, 2. front_profile, 3. radii_list,
        4. min_radius, 5. back_contour_index, 6. front_contour_index,
        7. contour_list, 8. rays_intensity_profile, 9. rays_indices, 10. selected_profile_index

        """
        if self._max_seed_correction_radius_in_image_coordinates:
            point = self._correct_seed(point)
        all_data = {"seed_corrected_point": point}
        radii_list = []
        min_radius = 100
        radius = 100
        contour_list = []
        rays_intensity_profile = []
        rays_indices = []

        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            front_ray_indices = self.get_ray_points_indices(point, phi, front=True)
            back_ray_indices = self.get_ray_points_indices(point, phi, front=False)

            ray_indices = construct_ray_from_half_rays(front_ray_indices, back_ray_indices, point)
            rays_indices.append(ray_indices)

            ray_intensity_profile = self.get_intensity_profile_from_ray_indices(ray_indices)
            rays_intensity_profile.append(ray_intensity_profile)

        all_data["rays_indices"] = rays_indices
        all_data["rays_intensity_profile"] = rays_intensity_profile

        for i, ray_indices in enumerate(rays_indices):

            ray_length = len(ray_indices)
            half_ray_length = (ray_length - 1) / 2

            back_contour_index = self.get_contour_index(point, list(reversed(ray_indices[0:half_ray_length + 1])))
            front_contour_index = self.get_contour_index(point, ray_indices[half_ray_length:ray_length])
            all_data["back_contour_index"] = back_contour_index
            all_data["front_contour_index"] = front_contour_index
            if len(back_contour_index) == 2 and len(front_contour_index) == 2:
                radius = tr.get_distance(back_contour_index, front_contour_index)
            contour_list.append([back_contour_index, front_contour_index])

            radii_list.append(radius)

            if radius < min_radius:
                all_data["min_radius"] = radius
                all_data["selected_ray_index"] = i

        # assert (min_radius < 100)
        contour_list.append(contour_list)
        all_data["contour_list"] = contour_list
        all_data["radii_list"] = radii_list
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
                intensity_value = 0.0
            profile_values.append(intensity_value)
        return profile_values

    def get_contour_index(self, point, ray_indices):

        image = self.image
        try:
            point_value = image.GetPixel([int(point[0]), int(point[1])])
        except RuntimeError as error:
            warnings.warn("Point outside the image! Assuming intensity = 0")
            point_value = 0.0

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

            if (front):
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
        center = point
        pixel_values = self.image.AbsImageFilter(self.circle(center))
        corrected_point = get_index_of_maximum(max(pixel_values))
        return corrected_point

    def circle(self, point):
        image_x = sitk.VectorIndexSelectionCast(self.image, 0)
        image_y = sitk.VectorIndexSelectionCast(self.image, 1)
        radius = self._max_seed_correction_radius_in_image_coordinates
        return (sitk.Sqrt((image_x - point[0]) ** 2 + (image_y - point[1]) ** 2)) < radius
