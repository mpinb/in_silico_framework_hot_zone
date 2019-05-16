
import numpy as np
import math

class RadiusCalculator:
    def __init__(self, xyResolution=0.092, zResolution=0.5, xySize=20,
                 numberOfRays=36, tresholdPercentage=0.5, numberOfRaysForPostMeasurment=20):
        self.xyResolution = xyResolution
        self.zResolution = zResolution
        self.xySize = xySize
        self.numberOfRays = numberOfRays
        self.rayLengthPerDirection = xySize / 2.0
        self.rayLengthPerDirectionOfImageCoordinates = self.rayLengthPerDirection/xyResolution
        self.tresholdPercentage = tresholdPercentage

        self.rayLengthPerDirectionOfImageCoordinatesForPostMeasurment = 0.50/xyResolution # self.rayLengthPerDirectionOfImageCoordinates/10
        self.numberOfRaysForPostMeasurment = numberOfRaysForPostMeasurment
        self.debug_postMeasurementPoints = []

    def getProfileOfThesePoints(self, image, points, postMeasurment='no'):
        temp = []
        AllPointsMinRadii = []

        for point in points:
            temp = self.getRadiusFromProfile(image, point, postMeasurment)
            AllPointsMinRadii.append(temp[3])

        return AllPointsMinRadii

    def getProfileAtThisPoint(self, image, point):
        raysProfiles = []
        rays = []
        for i in range(self.numberOfRays):
            phi = i*(np.pi/self.numberOfRays)

            frontCoordinates = self.getRayPointCoordinates(image, point, phi, front=True, postMeasurment='no')
            backCoordinates = self.getRayPointCoordinates(image, point, phi, front=False, postMeasurment='no')

            ray = self.constructRay(frontCoordinates, backCoordinates, point)
            rays.append(ray)




            rayProfile = self.getProfileValues(image, ray)
            raysProfiles.append(rayProfile)

        return rays, raysProfiles


    def getRadiusFromProfile(self, image, point, postMeasurment='no'):
        radiusList = []
        minRadius = 100
        radius = 100
        counterList = []
        raysProfileList = []
        backProfile = []
        frontProfile = []
        rays = []
        selectedProfileIndex = 1000
        backRadius = 0
        frontRadius = 0

        if postMeasurment == 'yes':
            rays, raysProfileList = self.getHigherResolutionProfiles(image, point)
            point = self.postMeasurmentFunction(image, rays, raysProfileList)
            self.debug_postMeasurementPoints.append(point)
            return self.getRadiusFromProfile(image, point, postMeasurment='no')


        for i in range(self.numberOfRays):
            phi = i*(np.pi/self.numberOfRays)

            frontCoordinate = self.getRayPointCoordinates(image, point, phi, front=True, postMeasurment='no')
            backCoordinate = self.getRayPointCoordinates(image, point, phi, front=False, postMeasurment='no')

            ray = self.constructRay(frontCoordinate, backCoordinate, point)
            rays.append(ray)

            rayProfile = self.getProfileValues(image, ray)
            raysProfileList.append(rayProfile)

        for i, ray in enumerate(rays):

            rayLength = len(ray)
            halfRayLength = (rayLength-1)/2

            backCounterPoint = self.getCounterIndex(image, point, list(reversed(ray[0:halfRayLength+1])))
            frontCounterPoint = self.getCounterIndex(image, point, ray[halfRayLength:rayLength])

            if (len(backCounterPoint) == 2 and len(frontCounterPoint) == 2):
                radius = self.getDistance(backCounterPoint,frontCounterPoint)
            counterList.append([backCounterPoint, frontCounterPoint])

            radiusList.append(radius)

            if (radius < minRadius):
                frontProfile = self.getProfileValues(image, frontCoordinate)
                backProfile = self.getProfileValues(image, backCoordinate)
                minRadius = radius
                selectedProfileIndex = i
        # assert (minRadius < 100)
        return backProfile, frontProfile, radiusList, minRadius, backCounterPoint, frontCounterPoint, counterList, raysProfileList, rays, selectedProfileIndex


    def getHigherResolutionProfiles(self, image, point):
        raysProfileList = []
        rays = []
        for i in range(self.numberOfRaysForPostMeasurment):
            phi = i*(np.pi/self.numberOfRaysForPostMeasurment)

            frontCoordinate = self.getRayPointCoordinates(image, point, phi, front=True, postMeasurment='yes')
            backCoordinate = self.getRayPointCoordinates(image, point, phi, front=False, postMeasurment='yes')

            ray = self.constructRay(frontCoordinate, backCoordinate, point)
            rays.append(ray)

            rayProfile = self.getProfileValues(image, ray)
            raysProfileList.append(rayProfile)

        return rays, raysProfileList

    def postMeasurmentFunction(self, image, rays, raysProfileList):

        maxIntensity = 0
        centerPoint = rays[0][(len(rays[0])-1)/2]
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
            profileValues.append(image.GetPixel(profileIndices[i]))
        return profileValues


    def getCounterIndex(self, image, point, profileIndices):
        pointValue = image.GetPixel([int(point[0]), int(point[1])])
        # pointHalfValue = pointValue/2.0
        pointTresholdValue = pointValue*self.tresholdPercentage

        profileIndicesLength = len(profileIndices)
        contourIndices = []
        for i in range(profileIndicesLength-1):
            pixel_1_value = image.GetPixel(profileIndices[i])
            pixel_2_value = image.GetPixel(profileIndices[i+1])

            if pixel_1_value >= pointTresholdValue and pixel_2_value <= pointTresholdValue:
                contourIndices = profileIndices[i]
                break

        return contourIndices


    def getDistance(self, point_1, point_2):
        return np.sqrt((point_1[0]-point_2[0])**2+(point_1[1]-point_2[1])**2)


    def getRayPointCoordinates(self, image, point, phi, front, postMeasurment='no'):

        if postMeasurment == 'no':
            rayLength = self.rayLengthPerDirectionOfImageCoordinates
        elif postMeasurment == 'yes':
            rayLength = self.rayLengthPerDirectionOfImageCoordinatesForPostMeasurment

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
