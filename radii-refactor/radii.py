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


class Radius_extractor:
    def __init__(self, points, image_file, xy_resolution=0.092, z_resolution=0.5, ray_length_front_to_back=20,
                 number_of_rays=36, threshold_percentage=0.5, max_seed_correction_radius_in_micron=20):
        """ This is the main method for extracting radii
        - Inputs:
            1. points: must be in the type transformation.coordinate_2d, so they are a list of
            2d values in micrometer
            2. image is a tif file with the relative resolution of xy_resolution to the points
             which are in micrometer

            3. xy_resolution: pixelSize in micrometers as acquired by the microscope in one optical section.
            4. z_resolution: z distance in micrometers between optical sections
            5. ray_length_front_to_back: maximum distance from the seed point considered in micrometer.

        """

        self.points = points  # TYPE: Must be transformation.coordinate_2d
        self.points_with_intensity = []
        self.recorded_seed_corrected_points = []
        self.seed_corrected_points_with_intensity = []
        self.image = self._read_image(image_file)
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution
        # ray_length_front_to_back, is the size of total ray in micron.
        self.ray_length_front_to_back = ray_length_front_to_back
        self.number_of_rays = number_of_rays
        self._ray_length_per_direction = ray_length_front_to_back / 2.0
        # __ray_length_per_direction_of_image_coordinates: by dividing _ray_length_per_direction with the
        # xy_resolution we go to the image coordinate system.
        self._ray_length_per_direction_of_image_coordinates = self._ray_length_per_direction / xy_resolution
        self.threshold_percentage = threshold_percentage
        self._max_seed_correction_radius_in_image_coordinates = max_seed_correction_radius_in_micron / xy_resolution
        self.contour_list = []
        self.profile_data = {}

    def get_intensity_profiles_by_point(self, point):
        """
        This method is useful when one need a profile of just one point in the image provided in at init.
        """
        image = self.image
        if self._max_seed_correction_radius_in_image_coordinates:
            point = self._correct_seed(point)
        rays_profiles = []
        rays = []
        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            front_coordinates = self.get_ray_point_coordinates(image, point, phi, front=True)
            back_coordinates = self.get_ray_point_coordinates(image, point, phi, front=False)

            ray = self.construct_ray(front_coordinates, back_coordinates, point)
            rays.append(ray)

            ray_profile = self.get_profile_values(image, ray)
            rays_profiles.append(ray_profile)

        return rays, rays_profiles

    def get_intensity_profile_by_points(self):
        """
        This is the main method of the class.
        To extract the radius of points from the image, after initiating the class, this method need to be called.

        """
        image = self.image
        points = self.points
        temp = []
        radii_from_all_points = []

        for point in points:
            temp = self.get_radius_from_profile(point)
            radii_from_all_points.append(temp[3])

        return radii_from_all_points

    def get_radius_from_profile(self, point):
        image = self.image
        profile_data = {}
        radius_list = []
        min_radius = 100
        radius = 100
        contour_list = []
        rays_profile_list = []
        back_profile = []
        front_profile = []
        rays = []
        selected_profile_index = 1000

        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            front_coordinate = self.get_ray_point_coordinates(image, point, phi, front=True, postMeasurment='no')
            back_coordinate = self.get_ray_point_coordinates(image, point, phi, front=False, postMeasurment='no')

            ray = self.construct_ray(front_coordinate, back_coordinate, point)
            rays.append(ray)

            ray_profile = self.get_profile_values(image, ray)
            rays_profile_list.append(ray_profile)

        for i, ray in enumerate(rays):

            ray_length = len(ray)
            half_ray_length = (ray_length - 1) / 2

            back_contour_point = self.get_counter_index(image, point, list(reversed(ray[0:half_ray_length + 1])))
            front_contour_point = self.get_counter_index(image, point, ray[half_ray_length:ray_length])
            profile_data["back_contour_point"] = back_contour_point
            profile_data["front_contour_point"] = front_contour_point
            if len(back_contour_point) == 2 and len(front_contour_point) == 2:
                radius = self.get_distance(back_contour_point, front_contour_point)
            contour_list.append([back_contour_point, front_contour_point])

            radius_list.append(radius)

            if radius < min_radius:
                profile_data["front_profile"] = self.get_profile_values(image, front_coordinate)
                profile_data["back_profile"] = self.get_profile_values(image, back_coordinate)
                profile_data["min_radius"] = radius
                profile_data["selected_profile_index"] = i
        # assert (min_radius < 100)
        self.contour_list.append(contour_list)
        self.profile_data["contour_list"] = self.contour_list
        return back_profile, front_profile, radius_list, min_radius, back_contour_point, front_contour_point, contour_list, rays_profile_list, rays, selected_profile_index

    def get_higher_resolution_profiles(self, image, point):
        rays_profile_list = []
        rays = []
        for i in range(self._max_seed_correction_radius_in_image_coordinates):
            phi = i * (np.pi / self._max_seed_correction_radius_in_image_coordinates)

            front_coordinate = self.get_ray_point_coordinates(image, point, phi, front=True, postMeasurment='yes')
            back_coordinate = self.get_ray_point_coordinates(image, point, phi, front=False, postMeasurment='yes')

            ray = self.construct_ray(front_coordinate, back_coordinate, point)
            rays.append(ray)

            ray_profile = self.get_profile_values(image, ray)
            rays_profile_list.append(ray_profile)

        return rays, rays_profile_list

    def post_measurment_function(self, rays, rays_profile_list):

        max_intensity = 0
        center_point = rays[0][(len(rays[0]) - 1) / 2]
        for idx, rayProfile in enumerate(rays_profile_list):

            index_of_max_value = np.argmax(np.array(rayProfile))
            new_max_intensity = rayProfile[index_of_max_value]

            floating_point = rays[idx][index_of_max_value]
            if new_max_intensity > max_intensity:
                max_intensity = new_max_intensity
                center_point = floating_point

        return center_point

    def construct_ray(self, front_profile_indices, back_profile_indices, point):
        center_point_index = [int(point[0]), int(point[1])]
        ray = list(reversed(back_profile_indices)) + [center_point_index] + front_profile_indices
        return ray

    def get_profile_values(self, image, profile_indices):
        profile_values = []
        profile_indices_length = len(profile_indices)
        for i in range(profile_indices_length):
            try:
                pixel = map(lambda x: (int(x)), profile_indices[i])
                intensity_value = image.GetPixel(pixel)
            except RuntimeError as error:
                warnings.warn(error)
                intensity_value = 0.0
            profile_values.append(intensity_value)
        return profile_values

    def get_counter_index(self, image, point, profile_indices):

        try:
            point_value = image.GetPixel([int(point[0]), int(point[1])])
        except RuntimeError as error:
            print(error)
            point_value = 0.0

        # pointHalfValue = point_value/2.0
        point_threshold_value = point_value * self.threshold_percentage

        profile_indices_length = len(profile_indices)
        contour_indices = []
        for i in range(profile_indices_length - 1):

            try:
                pixel_1_value = image.GetPixel(profile_indices[i])
            except IndexError as error:
                print(error)
                pixel_1_value = 0.0

            try:
                pixel_2_value = image.GetPixel(profile_indices[i + 1])
            except IndexError as error:
                print(error)
                pixel_2_value = 0.0

            if pixel_1_value >= point_threshold_value >= pixel_2_value:
                contour_indices = profile_indices[i]
                break

        return contour_indices

    def get_distance(self, point_1, point_2):
        return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

    def get_ray_point_coordinates(self, point, phi, front, postMeasurment='no'):
        image = self.image
        if postMeasurment == 'no':
            ray_length = self._ray_length_per_direction_of_image_coordinates
        elif postMeasurment == 'yes':
            ray_length = self.ray_length_per_direction_of_image_coordinates_for_post_measurement

        profile_indices = []

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
                profile_indices.append([int(x_new), int(y_new)])

        return profile_indices

    def _read_image(self, image_file):
        '''Reading image file '''
        image_file_reader = sitk.ImageFileReader()
        image_file_reader.SetFileName(image_file)
        image = image_file_reader.Execute()
        return image

    def _correct_seed(self, point):
        r = self._max_seed_correction_radius_in_image_coordinates
        corrected_point = f(r)
        return corrected_point


