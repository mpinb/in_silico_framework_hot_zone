import re


def hocFile(inputFilePath):
    with open(inputFilePath, 'r') as hocFile:
        lines = hocFile.readlines()
        neuron_section = False
        points = []

        for lineNumber, line in enumerate(lines):
            soma = line.rfind("soma")
            dend = line.rfind("dend")
            apical = line.rfind("apical")
            createCommand = line.rfind("create")
            pt3daddCommand = line.rfind("pt3dadd")

            if not neuron_section and ((createCommand > -1)
                                       and (soma + apical + dend > -3)):
                neuron_section = True

            if neuron_section and (line == '\n'):
                neuron_section = False

            if (pt3daddCommand > -1) and neuron_section:
                matches = re.findall('-?\d+\.\d+', line)
                point = map(float, matches)
                points.append(point)

    return points


