import re


def hocFile(inputFilePath):
    with open(inputFilePath, 'r') as hocFile:
        lines = hocFile.readlines()
        neuron_section = False
        points = []

        for lineNumber, line in enumerate(lines):
            soma = line.rfind("create")
            dend = line.rfind("dend")
            apical = line.rfind("apical")
            createCommand = line.rfind("create")
            pt3daddCommand = line.rfind("pt3dadd")

            if (createCommand > -1 & (soma + dend + apical > -3)):
                neuron_section = True
            elif (line == '\n'):
                neuron_section = False

            if (pt3daddCommand > -1 & neuron_section):
                matches = re.findall('-?\d+\.\d+', line)
                point = map(float, matches)
                points.append(point)

    return points


