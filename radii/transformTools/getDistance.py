import numpy as np


def nodes(points):

    class Edge:
        def __init__(self, distance, start, end):
            self.distance = distance
            self.start = start
            self.end = end

        def __str__(self):
            return str(self.__class__) + ": " + str(self.__dict__)

        def __repr__(self):
            return str(self.__class__) + ": " + str(self.__dict__)

    distances = []
    edges = []
    nodes = findNodes(points)

    for node_i in nodes:
        for node_j in nodes:
            tempDistance = distance(node_i, node_j)
            if (tempDistance not in distances):
                distances.append(tempDistance)
                tempEdge = Edge(tempDistance, node_i, node_j)
                edges.append(tempEdge)

    return edges


def findNodes(points):
    nodes = []
    for point in points:
        if (point not in nodes):
            if (points.count(point) > 1):
                nodes.append(point)
    return nodes


def distance(point_i, point_j):
    return np.sqrt((point_j[0]-point_i[0])**2 + (point_j[1]-point_i[1])**2 +
                   (point_j[2]-point_i[2])**2)


def longestEdge(edges):
    longestEdge = edges[0]
    for edge in edges:
        if (edge.distance > longestEdge.distance):
            longestEdge = edge
    return longestEdge


def matchDirection(set):

    centerOfSet1 = 0.0
    centerOfSet2 = 0.0
    length = len(set)

    for i in range(length):
        centerOfSet1 = centerOfSet1 + (np.array(set[i][0].start) +
                                       np.array(set[i][0].end))/length

        centerOfSet2 = centerOfSet2 + (np.array(set[i][1].start) +
                                       np.array(set[i][1].end))/length

    print("centerOfSet1 and Set2")
    print(centerOfSet1)
    print(centerOfSet1)

    for i in range(length):

        deltaSet1 = distance(centerOfSet1, set[i][0].start)\
            - distance(centerOfSet1, set[i][0].end)

        print("points from set 1")
        print(set[i][0].start)
        print(set[i][0].end)

        deltaSet2 = distance(centerOfSet2, set[i][1].start)\
            - distance(centerOfSet2, set[i][1].end)

        print("points from set 2")
        print(set[i][1].start)
        print(set[i][1].end)

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


def matchEdges(setA, setB, m):

    edgesA = nodes(setA)
    edgesB = nodes(setB)
    matchedSet = []

    for i in range(m):
        edgeA = longestEdge(edgesA)
        edgeB = longestEdge(edgesB)
        matchedSet.append([edgeA, edgeB])
        edgesA.remove(edgeA)
        edgesB.remove(edgeB)

    matchedSet = matchDirection(matchedSet)
    return matchedSet
