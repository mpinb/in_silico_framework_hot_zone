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

    return matchedSet
