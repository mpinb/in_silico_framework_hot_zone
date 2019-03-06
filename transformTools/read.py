import re


# Reading all neuronal points of a hoc file
def hocFileComplete(inputFilePath):
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


# Reading hoc file with only two points (top and bottom) from each section of
# neuronal points of a hoc file
def hocFileReduced(inputFilePath):
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


# Reading all points of am file with also their radius form the
# thickness part of the file
def amFile(inputFilePath):
    with open(inputFilePath, 'r') as amFile:
        lines = amFile.readlines()
        points = []
        rads = []
        pointsWithRad = []
        in_edge_section = False
        in_thickness_section = False

        for lineNumber, line in enumerate(lines):

            edge_sign = line.rfind("@8")
            EdgePointCoordinates = line.rfind("EdgePointCoordinates")

            thickness_sign = line.rfind("@9")
            thickness = line.rfind("thickness")

            if edge_sign > -1 and EdgePointCoordinates < 0:
                in_edge_section = True
                continue

            if thickness_sign > -1 and thickness < 0:
                in_thickness_section = True
                continue

            if in_edge_section and (line != '\n'):
                matches = re.findall('-?\d+\.\d+e[+-]?\d+', line)
                point = map(float, matches)
                points.append(point)

            if in_edge_section and (line == '\n'):
                in_edge_section = False

            if in_thickness_section and (line != '\n'):
                matches = re.findall('-?\d+\.\d+e[+-]?\d+', line)
                rad = map(float, matches)
                rads.append(rad)

            if in_thickness_section and (line == '\n'):
                in_thickness_section = False

        print(len(points))
        print(len(rads))
        for idx, point in enumerate(points):
            pointsWithRad.append([point[0], point[1], point[2], rads[idx][0]])

        return pointsWithRad
