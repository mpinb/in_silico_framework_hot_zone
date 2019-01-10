import re


def hocFile(inputFilePath):
    with open(inputFilePath, 'r') as hocFile:
        lines = hocFile.readlines()
        neuron_section = False
        points = []
        lastPoint = []

        in_neuron_line_number = 0

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
                in_neuron_line_number = 0
                points.append(lastPoint)
                lastPoint = []

            if (pt3daddCommand > -1) and neuron_section:
                in_neuron_line_number = in_neuron_line_number + 1;
                matches = re.findall('-?\d+\.\d+', line)
                point = map(float, matches)
                if (in_neuron_line_number == 1):
                    points.append(point)
                else:
                    lastPoint = point
    return points

