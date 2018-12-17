import numpy as np

def nodes(points):
    distances = []
    nodes = findNodes(points)
    for node_i in nodes:
        for node_j in nodes:
            distances.append(distance(node_i, node_j))
    return distances

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

