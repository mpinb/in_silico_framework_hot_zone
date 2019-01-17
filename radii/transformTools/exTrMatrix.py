import numpy as np
import re


def read(file):
    matrix = []
    vector = []
    row = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.rfind("TransformationMatrix")>-1:
                matches = re.findall('-?\d+\.\d+|-?\d+', line)
                vector = map(float, matches)
        for i in range(4):
            for j in range(4):
                k = j + i*4
                row.append(vector[k])
            matrix.append(row)
            row = []

    return matrix


def getTransformation(src, dst):
    x = np.transpose(np.matrix([src[0], src[1], src[2], src[3]]))
    y = np.transpose(np.matrix([dst[0], dst[1], dst[2], dst[3]]))

    # add ones on the bottom of x and y
    x = np.vstack((x,[1,1,1,1]))
    y = np.vstack((y,[1,1,1,1]))
    # solve for A2
    trMatrix = y * x.I

    return trMatrix

