# Please run this file in after using source_isf

import Interface as I
import re


# inputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/conFile.txt'
# inputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/egAm.txt'
inputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/amTransformed2.txt'
# outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/hocFile'
# outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/hocFile_reduced'
# outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/conFile_transformed'
# outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/egAm_landmark'
outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/am2_transformed_landmark'

points = []

with open(inputFile, 'r') as hocFile:
    lines = hocFile.readlines()
    for line in lines:
        matches = re.findall('-?\d+\.\d+', line)
        point = map(float, matches)
        points.append(point)

I.scp.write_landmark_file(outputFile, points)
