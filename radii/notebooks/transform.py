import os
import sys
import numpy as np

from scipy.spatial import ConvexHull

nb_dir = os.path.split(os.getcwd())[0]
if (nb_dir not in sys.path):
    sys.path.append(nb_dir)

import transformTools as tr
import radii as radi

hocDataPath = str('../data/neuron1/hoc/')
amDataPath = str('../data/neuron1/am_final/')
outputFolderPath = str('../output/neuron1/hoc/')

hocFile = hocDataPath + "500_GP_WR639_cell_1547_SP5C_checked_RE.hoc"
amFile = amDataPath + 'algined_slices_only_dendrite.am'

# extract set1 points

# remove radius from the set

pointsWithRadius = tr.read.hocFile(hocFile)
set1 = []
for el in pointsWithRadius:
    set1.append([el[0], el[1], el[2]])

# extract set2 points
points = radi.spacialGraph.getSpatialGraphPoints(amFile)

trMatrix = tr.exTrMatrix.read(amFile)

# conPoints = []
# convexSet2 = ConvexHull(points)

# for verIndex in convexSet2.vertices:
#     conPoints.append(points[verIndex])


# calculate transformation matrix:

set2 = points
# set2 = []
# for point in points:
#     point.append(1)
#     p = np.dot(np.array(point), trMatrix)
#     set2.append(p)


# conFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/conFile.txt'
# with open(conFile, 'w') as f:
#     for item in set2:
#         f.write('{:f}\t{:f}\t{:f} \n'.format(item[0], item[1], item[2]))


# find longest matched edges from set1 points and set2 points

numberOfEdges = 2

matchedSet = tr.getDistance.matchEdges(set1, set2, numberOfEdges)

# print(matchedSet)

egHocFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/egHoc.txt'
egAmFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/egAm.txt'

with open(egHocFile, 'w') as f:
    for item in matchedSet:
        startPoint = item[0].start
        endPoint = item[0].end
        f.write('{:f}\t{:f}\t{:f} \n'.format(startPoint[0], startPoint[1], startPoint[2]))
        f.write('{:f}\t{:f}\t{:f} \n'.format(endPoint[0], endPoint[1], endPoint[2]))


with open(egAmFile, 'w') as f:
    for item in matchedSet:
        startPoint = item[1].start
        endPoint = item[1].end
        f.write('{:f}\t{:f}\t{:f} \n'.format(startPoint[0], startPoint[1], startPoint[2]))
        f.write('{:f}\t{:f}\t{:f} \n'.format(endPoint[0], endPoint[1], endPoint[2]))

