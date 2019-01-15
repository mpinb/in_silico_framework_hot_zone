import numpy as np
from scipy.spatial import ConvexHull


def constructEdges(points):

    class Edge:
        def __init__(self, distance, start, end):
            self.distance = distance
            self.start = start
            self.end = end

        def __str__(self):
            return str(self.__class__) + ": " + str(self.__dict__)

        def __repr__(self):
            return str(self.__class__) + ": " + str(self.__dict__)

    nodes = points

    distances = []
    edges = []
    i = 0
    j = 0
    for node_i in nodes:
        for node_j in nodes:
            tempDistance = distance(node_i, node_j)
            if (tempDistance not in distances):
                distances.append(tempDistance)
                tempEdge = Edge(tempDistance, node_i, node_j)
                edges.append(tempEdge)
            j = j + 1
            print(i, j)
        i = i + 1
        j = 0
        print(i)
    return edges


def findHullPoints(points):
    conPoints = []
    convexSet = ConvexHull(points)

    for verIndex in convexSet.vertices:
        conPoints.append(points[verIndex])
    return conPoints


def findNodes(points):

    nodes = []
    for point in points:
        if (point not in nodes):
            if (points.count(point) > 1):
                nodes.append(point)

    print(nodes)
    return nodes


def distance(point_i, point_j):
    return np.sqrt((point_j[0]-point_i[0])**2 + (point_j[1]-point_i[1])**2 +
                   (point_j[2]-point_i[2])**2)


def longestOptimalEdges(edges):
    sortedEdges = sorted(edges, key=lambda x: x.distance, reverse=True)

    firstLongestEdge = sortedEdges[0]

    span = getMean(sortedEdges)/10

    isFound = False
    index = 1
    while (isFound is False):
        secondLongestEdge = sortedEdges[index]
        side1 = distance(secondLongestEdge.start, firstLongestEdge.start)
        side2 = distance(secondLongestEdge.end, firstLongestEdge.start)
        side3 = distance(secondLongestEdge.start, firstLongestEdge.end)
        side4 = distance(secondLongestEdge.end, firstLongestEdge.end)
        if (side1 > span and side2 > span and side3 > span and side4 > span):
            isFound = True
        index = index + 1

    return [firstLongestEdge, secondLongestEdge]


def matchDirection(set):

    centerOfSet1 = 0.0
    centerOfSet2 = 0.0
    length = len(set)

    for i in range(length):
        print(i)
        print("edg1:")
        print(set[i][0].start)
        print(set[i][0].end)
        print("edg2:")
        print(set[i][1].start)
        print(set[i][1].end)

        centerOfSet1 = centerOfSet1 + (np.array(set[i][0].start) +
                                       np.array(set[i][0].end))

        centerOfSet2 = centerOfSet2 + (np.array(set[i][1].start) +
                                       np.array(set[i][1].end))

    centerOfSet1 = centerOfSet1/length
    centerOfSet2 = centerOfSet2/length

    print("centerOfSet1 and Set2")
    print(centerOfSet1)
    print(centerOfSet2)

    for i in range(length):

        deltaSet1 = distance(centerOfSet1, set[i][0].start)\
            - distance(centerOfSet1, set[i][0].end)

        deltaSet2 = distance(centerOfSet2, set[i][1].start)\
            - distance(centerOfSet2, set[i][1].end)

        print("deltaSet1")
        print(deltaSet1)
        print("deltaSet2")
        print(deltaSet2)

        if (deltaSet1*deltaSet2 < 0):
            print("check if we go inside the if")
            temp = set[i][1].start
            set[i][1].start = set[i][1].end
            set[i][1].end = temp

    return set


def removeSameEdges(setEg, edg):
    setEg.remove(edg)
    print(setEg)
    for el in setEg:
        if (edg.start == el.start or edg.start == el.end or edg.end == el.start or edg.end == el.end):
            setEg.remove(el)
    return setEg


def getMean(set):
    mean = 0.
    for edg in set:
        mean = mean + edg.distance

    mean = mean / len(set)
    return mean


def matchEdges(setA, setB, m):

    hullPointsA = findHullPoints(setA)
    hullPointsB = findHullPoints(setB)

    edgesA = constructEdges(hullPointsA)
    edgesB = constructEdges(hullPointsB)

    matchedSet = []

    longestEdgesA = longestOptimalEdges(edgesA)
    longestEdgesB = longestOptimalEdges(edgesB)

    A_firstLongestEdg = longestEdgesA[0]
    A_secondLongestEdg = longestEdgesA[1]

    B_firstLongestEdg = longestEdgesB[0]
    B_secondLongestEdg = longestEdgesB[1]

    matchedSet.append([A_firstLongestEdg, B_firstLongestEdg])
    matchedSet.append([A_secondLongestEdg, B_secondLongestEdg])

    matchedSet = matchDirection(matchedSet)
    return matchedSet
