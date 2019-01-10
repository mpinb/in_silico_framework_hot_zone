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
