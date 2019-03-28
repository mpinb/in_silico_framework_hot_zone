
import numpy as np


class RadiusCalculator:
    def __init__(self, xyResolution=0.092, zResolution=0.5, xySize=20,
                 numberOfRays=36):
        self.xyResolution = xyResolution
        self.zResolution = zResolution
        self.xySize = xySize
        self.numberOfRays = numberOfRays
        self.rayLengthPerDirection = xySize / 2.0
        self.rayLengthPerDirectionOfImageCoordinates = self.rayLengthPerDirection/xyResolution

    def getProfileAtThisPoint(self, image, point):
        raysProfiles = []
        rays = []
        for i in range(self.numberOfRays):
            phi = i*(np.pi/self.numberOfRays)

            frontCoordinates = self.getRayPointCoordinates(image, point, phi, front=True)
            backCoordinates = self.getRayPointCoordinates(image, point, phi, front=False)

            ray = self.constructRay(frontCoordinates, backCoordinates, point)
            rays.append(ray)

            rayProfile = self.getProfileValues(image, ray)
            raysProfiles.append(rayProfile)

        return rays, raysProfiles


    def getRadiusFromProfile(self, image, point):
        radiusList = []
        minRadius = 100
        radius = 100
        counterList = []
        raysProfileList = []
        backProfile = []
        frontProfile = []
        for i in range(self.numberOfRays):
            phi = i*(np.pi/self.numberOfRays)

            frontProfileIndices = self.getRayPointCoordinates(image, point, phi, front=True)
            backProfileIndices = self.getRayPointCoordinates(image, point, phi, front=False)

            backCounterPoint = self.getCounterIndex(image, point, frontProfileIndices)
            frontCounterPoint = self.getCounterIndex(image, point, backProfileIndices)

            ray = self.constructRay(frontProfileIndices, backProfileIndices, point)

            rayProfile = self.getProfileValues(image, ray)
            raysProfileList.append(rayProfile)

            if (len(backCounterPoint) == 2 and len(frontCounterPoint) == 2):
                counterList.append([backCounterPoint, frontCounterPoint])
                radius = self.getDistance(backCounterPoint, frontCounterPoint)
                radiusList.append(radius)

            if (radius < minRadius):
                frontProfile = self.getProfileValues(image, frontProfileIndices)
                backProfile = self.getProfileValues(image, backProfileIndices)
                minRadius = radius
        return [backProfile, frontProfile, radiusList, minRadius, backCounterPoint, frontCounterPoint, counterList, raysProfileList]


    def constructRay(self, frontProfileIndices, backProfileIndices, point):
        centerPointIndex = [int(round(point[0])), int(round(point[1]))]
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
        pointHalfValue = pointValue/2.0

        profileIndicesLength = len(profileIndices)
        contourIndices = []
        for i in range(profileIndicesLength-1):
            pixel_1_value = image.GetPixel(profileIndices[i])
            pixel_2_value = image.GetPixel(profileIndices[i+1])

            if pixel_1_value >= pointHalfValue and pixel_2_value <= pointHalfValue:
                contourIndices = profileIndices[i]
                break

        return contourIndices


    def getDistance(self, point_1, point_2):
        return np.sqrt((point_1[0]-point_2[0])**2+(point_1[1]-point_2[1])**2)


    def getRayPointCoordinates(self, image, point, phi, front):
        profileIndices = []

        imageWidth = image.GetWidth()
        imageHeight = image.GetHeight()

        x_i = point[0]
        y_i = point[1]

        x_f = x_i
        y_f = y_i

        for index in range(int(self.rayLengthPerDirectionOfImageCoordinates)):

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
