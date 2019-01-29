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


def exRadSets(path_to_am, path_to_tif, path_to_output_folder):
    for spacialGraphFile in os.listdir(path_to_am):
        if spacialGraphFile.endswith(".am"):
            points = readPoints(path_to_am + spacialGraphFile)
            spacialGraphIndicator = re.findall(r'[sS]\d+', spacialGraphFile)[0]
            outputFile = path_to_output_folder + spacialGraphIndicator + \
                "_with_r" + ".am"
            for imageFile in os.listdir(path_to_tif):
                if imageFile.startswith(spacialGraphIndicator):
                    image = readImage(path_to_tif + imageFile)
                    result = radi.radius.getRadiiHalfMax(image, points)
                    print(imageFile)
                    writeResult(path_to_am + spacialGraphFile, outputFile,
                                result)
                    break


def readImage(imageFile):
    imageFileReader = sitk.ImageFileReader()
    imageFileReader.SetFileName(imageFile)
    image = imageFileReader.Execute()
    return image


def readPoints(dataFile):
    points = radi.spacialGraph.getSpatialGraphPoints(dataFile)
    points = list(map(lambda x: map(lambda y: int(y/0.092), x), points))
    return points


def writeResult(inputDataFile, outputDataFile, result):
    radii = result[1]
    radii = [r*0.092 for r in radii]
    radi.spacialGraph.write_spacial_graph_with_thickness(inputDataFile,
                                                         outputDataFile, radii)
