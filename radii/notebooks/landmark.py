# Please run this file in after using source_isf

import Interface as I
import re


inputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/conFile.txt'
# outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/hocFile'
# outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/hocFile_reduced'
outputFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/conFile_transformed'

points = []

with open(inputFile, 'r') as hocFile:
    lines = hocFile.readlines()
    for line in lines:
        matches = re.findall('-?\d+\.\d+', line)
        point = map(float, matches)
        points.append(point)

I.scp.write_landmark_file(outputFile, points)
