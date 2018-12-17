import numpy as np
import os
import re


def hocFile(inputFilePath, outputFilePath):
    with open(inputFilePath, 'r') as hocFile:
        lines = hocFile.readlines
        neuron_section = False

        for lineNumber, line in enumerate(lines):
            soma = line.rfind("create")
            dend = line.rfind("dend")
            apical = line.rfind("apical")
            createCommand = line.rfind("create")
            pt3daddCommand = line.rfind("pt3dadd")
            pt3dclearCommand = line.rfind("pt3dclear()")

            if (createCommand & (soma | dend | apical)):
                neuron_section = True
            if (pt3daddCommand > -1):
                print(line[0])
                print(line[2])
                print(line[7])
                print(line[10])
