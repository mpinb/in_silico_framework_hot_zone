import os
import sys
import numpy as np
import time

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
hocFileOutput = outputFolderPath + "hocFileWithRad.hoc"
amFile = amDataPath + 'algined_slices_only_dendrite.am'

# extract set1 points

# remove radius from the set

pointsWithRadius = tr.read.hocFileReduced(hocFile)
set1 = []
for el in pointsWithRadius:
    set1.append([el[0], el[1], el[2]])
# extract set2 points
points = radi.spacialGraph.getSpatialGraphPoints(amFile)
set2 = points

# trMatrix = tr.exTrMatrix.read(amFile)

# conPoints = []
# convexSet2 = ConvexHull(points)

# for verIndex in convexSet2.vertices:
#     conPoints.append(points[verIndex])


# calculate transformation matrix:

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

print("in the matchSet process")
matchedSet = tr.getDistance.matchEdges(set1, set2, numberOfEdges)

src = []
dst = []
for point in matchedSet:
    src.append(point[0].start)
    dst.append(point[1].start)

    src.append(point[0].end)
    dst.append(point[1].end)

print("In the calculations of the transofrmation matrix")
trMatrix2 = tr.exTrMatrix.getTransformation(dst, src)


amPoints4D = tr.read.amFile(amFile)


print("Applying the transofrmation matrix to the initial am points")
trAmPoints4D = []
for point4D in amPoints4D:
    point = point4D[:3]
    point.append(1)
    p = np.dot(trMatrix2, np.array(point))
    p_listed = p.tolist()[0]
    trAmPoints4D.append([p_listed[0], p_listed[1], p_listed[2], point4D[3]])

print(amPoints4D[0])
print(trAmPoints4D[0])

print(amPoints4D[10])
print(trAmPoints4D[10])

print(amPoints4D[1234])
print(trAmPoints4D[1234])
print(amPoints4D[54673])
print(trAmPoints4D[54673])
print(amPoints4D[11])
print(trAmPoints4D[11])
print(amPoints4D[56])
print(trAmPoints4D[56])

# amPoints = []
# for point in set2:
#     point.append(1)
#     p = np.dot(trMatrix2, np.array(point))
#     amPoints.append(p)

# transformedPoints = []
# for it in amPoints:
#     transformedPoints.append([it.item(0,0), it.item(0,1), it.item(0,2)])


# Tested
print("reading hoc file completely")
hocPointsComplete = tr.read.hocFileComplete(hocFile)
hocSet = []
for el in hocPointsComplete:
    hocSet.append([el[0], el[1], el[2]])

print("in the process of converting list of numpy array to list of lists")
# trAmPoints4DList = [p.tolist()[0] for p in trAmPoints4D]
trAmPoints4DList = trAmPoints4D
print(trAmPoints4DList[0])
print(amPoints4D[0])

print(trAmPoints4DList[2])
print(amPoints4D[2])

print(trAmPoints4DList[100])
print(amPoints4D[100])

print(trAmPoints4DList[10])
print(amPoints4D[10])

print(trAmPoints4DList[27685])
print(amPoints4D[27685])

print(trAmPoints4DList[17892])
print(amPoints4D[17892])

print(trAmPoints4DList[1293])
print(amPoints4D[1293])



print("In the process of finding pairs in between hoc file and the transoformed points to add radi to hocpoint")

startTime = time.time()
pairs = radi.addRadii.findPairs(trAmPoints4DList, hocSet)
endTime = time.time()
print(endTime - startTime)


# Little Test of the function above

# print(hocSet[0])
# print(pairs[0][0])
# print("---------")
# print(hocSet[1])
# print(pairs[1][0])
# print("---------")
# print(hocSet[42])
# print(pairs[42][0])
# print("---------")
# print(hocSet[13099])
# print(pairs[13099][0])
# print("---------")
# print(hocSet[13100])
# print(pairs[13100][0])
# print("---------")

initialAmPointsWithRad = tr.read.amFile(amFile)
initialAmPointsWithRad2 = radi.spacialGraph.getSpatialGraphPoints(amFile)



# Little Test of the function above

print(initialAmPointsWithRad[0])
print(initialAmPointsWithRad2[0])
print("---------")
print(initialAmPointsWithRad[1])
print(initialAmPointsWithRad2[1])
print("---------")
print(initialAmPointsWithRad[42])
print(initialAmPointsWithRad2[42])
print("---------")
print(initialAmPointsWithRad[len(initialAmPointsWithRad)-1])
print(initialAmPointsWithRad2[len(initialAmPointsWithRad2)-1])

startTime = time.time()
hocWithRad = []
newHPoint = []

print("adding radii to corresponding hoc point, based on the pairs")

outTestFolder = str('../output/neuron1/test/')

pairesFile = "pairs.txt"
trAmPoints4DFile = "trAmPoints4D.txt"
amPoints4DFile = "amPoints4D.txt"

testPairs = outTestFolder + pairesFile
testTrAmPoints4D = outTestFolder + trAmPoints4DFile
testAmPoints4D = outTestFolder + amPoints4DFile

with open(testPairs, 'w') as f:
    for pai in pairs:
        f.write('[[{:f},{:f},{:f},{:f}],\t[{:f},{:f},{:f}]] \n'.format(pai[0][0],pai[0][1],pai[0][2],pai[0][3], pai[1][0], pai[1][1], pai[1][2]))

with open(testTrAmPoints4D, 'w') as f:
    for tp in trAmPoints4DList:
        f.write('[{:f},{:f},{:f},{:f}] \n'.format(tp[0],tp[1],tp[2],tp[3]))

with open(testAmPoints4D, 'w') as f:
    for ap in amPoints4D:
        f.write('[{:f},{:f},{:f},{:f}] \n'.format(ap[0],ap[1],ap[2],ap[3]))

for pair in pairs:
    newHPoint = pair[1]
    dual = pair[0]
    newHPoint.append(dual[3])
    hocWithRad.append(newHPoint)
    newHPoint = []

# for p in pairs:
#     for idx, it in enumerate(transformedPoints):
#         hocPointNoRad = p[0]
#         if it[0]==p[1][0] and it[1]==p[1][1] and it[2]==p[1][2]:
#             hocPointNoRad.append(initialAmPointsWithRad[idx][3][0])
#             hocWithRad.append(hocPointNoRad)
#             continue
endTime = time.time()
print(endTime - startTime)


# Little Test of the function above

print(hocSet[0])
print(hocWithRad[0])
print("---------")
print(hocSet[1])
print(hocWithRad[1])
print("---------")
print(hocSet[42])
print(hocWithRad[42])
print("---------")
print(hocSet[13099])
print(hocWithRad[13099])
print("---------")
print(hocSet[13100])
print(hocWithRad[13100])
print("---------")

print("writing the final result in the output hocFile")

startTime = time.time()
tr.write.hocFile(hocFile, hocFileOutput, hocWithRad)
endTime = time.time()
print(endTime - startTime)

# print(list(newPoints[3]))
# print(newPoints[1])
# for it in newPoints:
#     print(it.item((0,3)))

# egHocFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/egHoc.txt'
# egAmFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/egAm.txt'

# with open(egHocFile, 'w') as f:
#     for item in matchedSet:
#         startPoint = item[0].start
#         endPoint = item[0].end
#         f.write('{:f}\t{:f}\t{:f} \n'.format(startPoint[0], startPoint[1], startPoint[2]))
#         f.write('{:f}\t{:f}\t{:f} \n'.format(endPoint[0], endPoint[1], endPoint[2]))


# with open(egAmFile, 'w') as f:
#     for item in matchedSet:
#         startPoint = item[1].start
#         endPoint = item[1].end
#         f.write('{:f}\t{:f}\t{:f} \n'.format(startPoint[0], startPoint[1], startPoint[2]))
#         f.write('{:f}\t{:f}\t{:f} \n'.format(endPoint[0], endPoint[1], endPoint[2]))


amTrFile = '/home/amir/Projects/radii/radii/data/neuron1/landmark/amTransformed2.txt'
with open(amTrFile, 'w') as f:
    for it in trAmPoints4DList:
        f.write('{:f}\t{:f}\t{:f} \n'.format(it[0], it[1], it[2]))

