# This module extracts all of the radii sets, each set corresponds to an
# .am file, and writes the calculated radii in new .am files in a new folder.

# Inputs:
# path_to_am: path to the folder contains the initial .am files without radius.
# path_to_tif: path to the folder contains the tif images corresponds to .am
# files.
# path_to_output: path to the output folder and its name.

# Outputs:
# 1. calculats radii sets, and puts each set inside its corresponding .am file
# inside the given output folder path.

import os
import re
import radii as radi
import SimpleITK as sitk

class RadiusCalculatorForManyFiles:
    def __init__(self, xyResolution=0.092, zResolution=0.5, xySize=20,
                 numberOfRays=10, tresholdPercentage=0.5, numberOfRaysForPostMeasurment=20):
        self.xyResolution = xyResolution
        self.zResolution = zResolution
        self.xySize = xySize
        self.numberOfRays = numberOfRays
        self.tresholdPercentage = tresholdPercentage

        self.rayLengthPerDirectionOfImageCoordinatesForPostMeasurment = 0.50/xyResolution # self.rayLengthPerDirectionOfImageCoordinates/10
        self.numberOfRaysForPostMeasurment = numberOfRaysForPostMeasurment

        self.radiusCalculator = radi.calcRad.RadiusCalculator(xyResolution=self.xyResolution, zResolution=self.zResolution, xySize=self.xySize, numberOfRays=self.numberOfRays, tresholdPercentage=self.tresholdPercentage)


    # extraxt radii sets of bunch of files from the folder of "path_to_am"
    # and writ them to and output folder
    def exRadSets(self, path_to_am, path_to_tif, path_to_output_folder, postMeasurment='no'):
        if (os.path.isdir(path_to_am) and os.path.isdir(path_to_tif)):
            for spacialGraphFile in os.listdir(path_to_am):
                if spacialGraphFile.endswith(".am"):
                    points = self.readPoints(path_to_am + spacialGraphFile)
                    spacialGraphIndicator = re.findall(r'[sS]\d+', spacialGraphFile)[0]
                    outputFile = path_to_output_folder + spacialGraphIndicator + \
                        "_with_r" + ".am"
                    for imageFile in os.listdir(path_to_tif):
                        if imageFile.startswith(spacialGraphIndicator):
                            image = self.readImage(path_to_tif + imageFile)
                            # result = radi.radius.getRadiiHalfMax(image, points)
                            result = self.radiusCalculator.getProfileOfThesePoints(image, points, postMeasurment)
                            print(imageFile)
                            self.writeResult(path_to_am + spacialGraphFile, outputFile, result)
                            break
        else:
            points = self.readPoints(path_to_am)
            amFileName = os.path.basename(path_to_am)
            outputFile = path_to_output_folder + "/" + amFileName
            imageFile = path_to_tif
            image = self.readImage(imageFile)
            result = self.radiusCalculator.getProfileOfThesePoints(image, points, postMeasurment)
            print(imageFile)
            self.writeResult(path_to_am, outputFile, result)


    # reading image file
    def readImage(self, imageFile):
        imageFileReader = sitk.ImageFileReader()
        imageFileReader.SetFileName(imageFile)
        image = imageFileReader.Execute()
        return image

    # return points of a am file, by using the function "getSpatialGraphPoints"
    def readPoints(self, dataFile):
        points = radi.spacialGraph.getSpatialGraphPoints(dataFile)
        points = list(map(lambda x: map(lambda y: int(y/0.092), x), points))
        return points

    # This function will write the result of the extracted radii to final am file
    def writeResult(self, inputDataFile, outputDataFile, result):
        radii = result
        radii = [r*0.092 for r in radii]
        radi.spacialGraph.write_spacial_graph_with_thickness(inputDataFile, outputDataFile, radii)
