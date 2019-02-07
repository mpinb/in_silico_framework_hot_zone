import numpy as np
import math
from scipy.spatial import ConvexHull

# we give this function a set of points and it will reaturn us all edges,
# that is possible to create from those points;
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
        i = i + 1
        j = 0
    return edges


# Find convex hull points from a set 3d points
def findHullPoints(points):
    conPoints = []
    convexSet = ConvexHull(points)

    for verIndex in convexSet.vertices:
        conPoints.append(points[verIndex])
    return conPoints


# Calculate the distance of two points, specially for calculating
# the length of an edge
def distance(point_i, point_j):
    dist = math.sqrt((point_j[0]-point_i[0])**2 + (point_j[1]-point_i[1])**2 +
                   (point_j[2]-point_i[2])**2)
    return dist


# here we sort the edges from the set of edges based on their length,
# we do not consider some edges, like the one that
# their start or end points are so close to another edge's start or end points
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


# In this function we need to match the direction of edges to each other
# Suppos we have two edges that we know their are corresponding to
# each other between two mythologies but we do not know if in which direction.
# So we give them to this function, and this function will find the direction
# and correct it in the set. It works by finding the distance of each point of
# the edge from their corresponding morphology's center of mass,
# then comparing them together.
# Example: Suppose we have two edges of A and B. A and B each contains two
# points which we arbitrary will call them A_start, A_end, B_start, B_end.
# Now we calculate the center of mass of each morphology, suppose they are
# Center_A and Center_B. Since the edges A and B are corresponds to each other,
# thus, If for example point A_start is corresponds to point B_start and the
# distance of A_start to center_a is less than the distance of
# A_end to center_end, THEN the distance of B_start to center_b must also be
# less than the distance of B_end to center_b, otherwise we need to swap the
# points of B_start with B_end with each other to be match with A_start and A_end
def matchDirection(set):

    centerOfSet1 = 0.0
    centerOfSet2 = 0.0
    length = len(set)

    for i in range(length):

        centerOfSet1 = centerOfSet1 + (np.array(set[i][0].start) +
                                       np.array(set[i][0].end))

        centerOfSet2 = centerOfSet2 + (np.array(set[i][1].start) +
                                       np.array(set[i][1].end))

    centerOfSet1 = centerOfSet1/length
    centerOfSet2 = centerOfSet2/length

    for i in range(length):

        deltaSet1 = distance(centerOfSet1, set[i][0].start)\
            - distance(centerOfSet1, set[i][0].end)

        deltaSet2 = distance(centerOfSet2, set[i][1].start)\
            - distance(centerOfSet2, set[i][1].end)

        if (deltaSet1*deltaSet2 < 0):
            temp = set[i][1].start
            set[i][1].start = set[i][1].end
            set[i][1].end = temp

    return set


# We do not use this function in the code, but this function will remove
# repeated edges in the set, We skip this function since,
# the function "longestOptimalEdges" Will guaranty to not have such a problem
def removeSameEdges(setEg, edg):
    setEg.remove(edg)
    for el in setEg:
        if (edg.start == el.start or edg.start == el.end or edg.end == el.start or edg.end == el.end):
            setEg.remove(el)
    return setEg

# Calculate the avarege length of all edges in an edges set
def getMean(set):
    mean = 0.
    for edg in set:
        mean = mean + edg.distance

    mean = mean / len(set)
    return mean


# Function below will find mateched points between two sets of points
# the parameter m is indicating how many edges we want to match,
# notice each edges have two points
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
