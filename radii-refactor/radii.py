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

    def get_intensity_profiles_by_point(self, image, point):
        if self._max_seed_correction_radius_in_image_coordinates:
            point = self._correct_seed(point)
        rays_profiles = []
        rays = []
        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            front_coordinates = self.get_ray_point_coordinates(image, point, phi, front=True)
            back_coordinates = self.get_ray_point_coordinates(image, point, phi, front=False)

            ray = self.constructRay(front_coordinates, back_coordinates, point)
            rays.append(ray)

            ray_profile = self.getProfileValues(image, ray)
            rays_profiles.append(ray_profile)

        return rays, rays_profiles

    def get_intensity_profile_by_points(self, image, points):
        temp = []
        radii_from_all_points = []

        for point in points:
            temp = self.get_radius_from_profile(image, point)
            radii_from_all_points.append(temp[3])

        return radii_from_all_points

    def get_radius_from_profile(self, image, point):
        profile_data = {}
        radius_list = []
        minRadius = 100
        radius = 100
        contour_list = []
        raysProfileList = []
        backProfile = []
        frontProfile = []
        rays = []
        selectedProfileIndex = 1000

        for i in range(self.number_of_rays):
            phi = i * (np.pi / self.number_of_rays)

            frontCoordinate = self.get_ray_point_coordinates(image, point, phi, front=True, postMeasurment='no')
            backCoordinate = self.get_ray_point_coordinates(image, point, phi, front=False, postMeasurment='no')

            ray = self.constructRay(frontCoordinate, backCoordinate, point)
            rays.append(ray)

            rayProfile = self.getProfileValues(image, ray)
            raysProfileList.append(rayProfile)

        for i, ray in enumerate(rays):

            rayLength = len(ray)
            halfRayLength = (rayLength - 1) / 2

            back_contour_point = self.getCounterIndex(image, point, list(reversed(ray[0:halfRayLength + 1])))
            front_contour_point = self.getCounterIndex(image, point, ray[halfRayLength:rayLength])
            profile_data["back_contour_point"] = back_contour_point
            profile_data["front_contour_point"] = front_contour_point
            if len(back_contour_point) == 2 and len(front_contour_point) == 2:
                radius = self.getDistance(back_contour_point, front_contour_point)
            contour_list.append([back_contour_point, front_contour_point])

            radius_list.append(radius)

            if (radius < minRadius):
                profile_data["frontProfile"] = self.getProfileValues(image, frontCoordinate)
                profile_data["backProfile"] = self.getProfileValues(image, backCoordinate)
                profile_data["minRadius"] = radius
                profile_data["selectedProfileIndex"] = i
        # assert (minRadius < 100)
        self.contour_list.append(contour_list)
        self.profile_data["contour_list"] = self.contour_list
        return backProfile, frontProfile, radius_list, minRadius, back_contour_point, front_contour_point, contour_list, raysProfileList, rays, selectedProfileIndex

    def getHigherResolutionProfiles(self, image, point):
        raysProfileList = []
        rays = []
        for i in range(self._max_seed_correction_radius_in_image_coordinates):
            phi = i * (np.pi / self._max_seed_correction_radius_in_image_coordinates)

            frontCoordinate = self.get_ray_point_coordinates(image, point, phi, front=True, postMeasurment='yes')
            backCoordinate = self.get_ray_point_coordinates(image, point, phi, front=False, postMeasurment='yes')

            ray = self.constructRay(frontCoordinate, backCoordinate, point)
            rays.append(ray)

            rayProfile = self.getProfileValues(image, ray)
            raysProfileList.append(rayProfile)

        return rays, raysProfileList

    def postMeasurmentFunction(self, image, rays, raysProfileList):

        maxIntensity = 0
        centerPoint = rays[0][(len(rays[0]) - 1) / 2]
        for idx, rayProfile in enumerate(raysProfileList):

            indexOfMaxValue = np.argmax(np.array(rayProfile))
            newMaxIntensity = rayProfile[indexOfMaxValue]

            floatingPoint = rays[idx][indexOfMaxValue]
            if newMaxIntensity > maxIntensity:
                maxIntensity = newMaxIntensity
                centerPoint = floatingPoint

        return centerPoint

    def constructRay(self, frontProfileIndices, backProfileIndices, point):
        centerPointIndex = [int(point[0]), int(point[1])]
        ray = list(reversed(backProfileIndices)) + [centerPointIndex] + frontProfileIndices
        return ray

    def getProfileValues(self, image, profileIndices):
        profileValues = []
        profileIndicesLength = len(profileIndices)
        for i in range(profileIndicesLength):
            try:
                pixel = map(lambda x: (int(x)), profileIndices[i])
                intensityValue = image.GetPixel(pixel)
            except RuntimeError as error:
                warnings.warn(error)
                intensityValue = 0.0
            profileValues.append(intensityValue)
        return profileValues

    def getCounterIndex(self, image, point, profileIndices):

        try:
            pointValue = image.GetPixel([int(point[0]), int(point[1])])
        except RuntimeError as error:
            print(error)
            pointValue = 0.0

        # pointHalfValue = pointValue/2.0
        pointTresholdValue = pointValue * self.threshold_percentage

        profileIndicesLength = len(profileIndices)
        contourIndices = []
        for i in range(profileIndicesLength - 1):

            try:
                pixel_1_value = image.GetPixel(profileIndices[i])
            except IndexError as error:
                print(error)
                pixel_1_value = 0.0

            try:
                pixel_2_value = image.GetPixel(profileIndices[i + 1])
            except IndexError as error:
                print(error)
                pixel_2_value = 0.0

            if pixel_1_value >= pointTresholdValue and pixel_2_value <= pointTresholdValue:
                contourIndices = profileIndices[i]
                break

        return contourIndices

    def getDistance(self, point_1, point_2):
        return np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)

    def get_ray_point_coordinates(self, image, point, phi, front, postMeasurment='no'):

        if postMeasurment == 'no':
            rayLength = self._ray_length_per_direction_of_image_coordinates
        elif postMeasurment == 'yes':
            rayLength = self.ray_length_per_direction_of_image_coordinates_for_post_measurement

        profileIndices = []

        imageWidth = image.GetWidth()
        imageHeight = image.GetHeight()

        x_i = point[0]
        y_i = point[1]

        x_f = x_i
        y_f = y_i

        for index in range(int(rayLength)):

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

            if (x_new <= 1 or y_new <= 1 or x_new >= imageWidth or y_new >= imageHeight):
                break
            else:
                profileIndices.append([int(x_new), int(y_new)])

        return profileIndices

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


