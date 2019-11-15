"""
IO Module

==========
This module contains methods to read and write data with different formats.


File extensions
----------------------

1. .am
Capabilities:
- Reading  all data point with their associated attribute (e.g. VertexLabels, thickness or radius)
- Writing data points in 3d and their corresponding radius
Limitations:
- Not able to read other data than 3d positional points and their radius
- Only able to read and write in ascii format (In this module in the class amira_utils the are methods one can use to
 convert amira binary format to ascii format, for more details look at the amira_utils class docstring)

2 .hoc
Capabilities:
- Reading data points (point coordinates and their associated radius) of kind:
    1. ApicalDendrite
    2. BasalDendrite
    3. Dendrite
    4. Soma
from the hoc file.
- Writing data points (as kind of what it can read) in 3d and their associated radius
Limitations:
- Not able to construct the tree structure from the hoc file.
- Not able to read the white matter data.

3 .hx
Capabilities:
- Read transformation matrix (the one for complete morphology not
each transformation for each file.) and the path files of the each slice.
Limitations:
- To read the transformations of each slice, it needs to connect to Amira.

Tests
-----

- The test functions are inside the test.py. One can also use them as example of how to use the functions.

"""
import Interface as I
import re
import os
from random import randrange


class Am:

    def __init__(self, input_path, output_path=None):

        if output_path is None:
            output_path = os.path.dirname(input_path) + "output_" + str(randrange(100))

        self.commands = {}
        self.config = {}
        self.input_path = input_path
        self.output_path = output_path
        self.all_data = {}

    def read(self):
        """
        Reading all data of of the am file

        """
        with open(self.input_path, 'r') as f:
            self.commands, config_end = self._read_config_and_commands()
            lines = f.readlines()
            for cs in self.commands:
                command_sign = self.commands[cs]
                # command_sign (eg. @1 or @2 ) are the initialized value of commands dict keys
                # which provided by the _read_commands function.
                data_section = False
                data = []
                for line in lines[config_end:]:
                    if line.rfind(command_sign) > -1:
                        data_section = True
                        continue
                    if data_section and line != '\n':
                        d = _read_numbers_in_line(line)
                        data.append(d)
                    elif data_section and line == '\n':
                        data_section = False
                self.all_data[cs] = data
        return self.all_data

    def write(self, input_path=None, output_path=None, all_data=None):
        """
        Writing data from a dictionary into an am file.
        """
        if input_path is None:
            input_path = self.input_path
        if output_path is None:
            output_path = self.output_path
        if all_data is None:
            all_data = self.all_data
        self._write_from_dict()

    def _write_from_dict(self):
        with open(self.output_path, "w") as data_file:
            data_file.writelines(self.all_data["config"])
            for cs in self.commands:
                data_file.write("\n")
                data_file.write(self.commands[cs])
                data_file.write("\n")
                for data in self.all_data[cs]:
                    string = ' '.join(map(str, data))
                    for item in string:
                        data_file.write(item)
                    data_file.write("\n")

    def _read_config_and_commands(self):
        with open(self.input_path, 'r') as fc:
            lines = fc.readlines()
            commands = {}
            config_end = 0
            for idx, line in enumerate(lines):
                if line.rfind("@") > -1:
                    # command_sign supposes to hold the values like @1 or @2 or ...
                    command_sign = "@" + line[line.rfind("@") + 1:].strip()
                    if line.replace(command_sign, "").strip() != "":
                        commands[line.replace(command_sign, "").strip()] = command_sign
                    else:
                        config_end = idx
                        break
            self.all_data["config"] = lines[:config_end]
        return commands, config_end


class Hoc:

    def __init__(self, input_path, output_path=None):

        if output_path is None:
            output_path = os.path.dirname(input_path) + "output_" + str(randrange(100))
        self.output_path = output_path
        self.input_path = input_path
        self.edges = I.scp.reader.read_hoc_file(input_path)
        self.all_data = {}
        self._process()

    def _process(self):
        self._extract_nodes()
        self._extract_all_pts()

    def _extract_nodes(self):
        nodes = []
        for e in self.edges:
            nodes.append(e.edgePts[0])
            nodes.append(e.edgePts[-1])
        self.all_data['nodes'] = nodes

    def _extract_all_pts(self):
        pts = []
        for e in self.edges:
            pts.extend(e.edgePts)
        self.all_data['all_points'] = pts

    def update_radii(self, radii, input_path=None, output_path=None):
        """
        # Writing radius of points to a specific hoc file.
        # basically it do this: reading a file without the
        # radii of neuronal points and add the radius to them in another hoc file
        
        Inputs:
        - 1. radii: A list of radii, which are floats values, the order of 
        radii list must be match with the oder of self.all_data["points"]
        
        - 2. input_path, if not given it will use self.input_path, the method will use this 
        as a sample hoc file to create another Hoc file with radii added to the corresponding points 
        
        - 3. output_path: The path of the desired output hoc file. If not given, the method will use self.output_path.   
        - 3. output_path: The path of the desired output hoc file. If not given, the method will use self.output_path.
        """

        if input_path is None:
            input_path = self.input_path
        if output_path is None:
            output_path = self.output_path

        with open(input_path, 'r') as readHocFile:
            with open(output_path, 'w') as writeHocFile:
                lines = readHocFile.readlines()
                neuron_section = False

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

                    if (pt3daddCommand > -1) and neuron_section:

                        hocPoint = radii[in_neuron_line_number]

                        line = line.replace("pt3dadd", "")
                        matches = re.findall('-?\d+\.\d?\d+|\-?\d+', line)
                        point = map(float, matches)

                        writeHocFile.write('{{pt3dadd({:f},{:f},{:f},{:f})}}\n'.format(hocPoint[0],
                                                                                       hocPoint[1],
                                                                                       hocPoint[2],
                                                                                       hocPoint[3]))
                        in_neuron_line_number = in_neuron_line_number + 1
                    else:
                        writeHocFile.write(line)


class Amira_utils:

    def __init__(self):
        pass


def _read_numbers_in_line(line):
    """
    Find numbers of in a line, the matches is a list contains
    the numbers that the regex command matches in the line.
    The number formats that this regex support are as an examples:
    egs:
    - 12 -> 12.0
    - -12 -> -12.0
    - 1.22 -> 1.22
    - -1.22 -> -1.22
    - 2.407640075683594e+02  -> 2.407640075683594e+02
    - -2.407640075683594e+02 -> -2.407640075683594e+02
    - -2.407640075683594e-02 -> -2.407640075683594e-02
    - 2.407640075683594e-02 -> 2.407640075683594e-02
    - -2.407640075683594 -> -2.407640075683594
    - 2.521719970703125e+02, 3.437120056152344e+02, 6.554999947547913e-01, -> 2.521719970703125e+02
    3.437120056152344e+02 6.554999947547913e-01

    """
    matches = re.findall('-?\d+\.\d+[e]?[+-]?\d+|\-?\d+[e]?', line)
    if not matches:
        raise RuntimeError("Expected number in line {} but did not find any".format(line))
    data = map(float, matches)
    return data
