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
        data = []
        with open(self.input_path, 'r') as f:
            self.commands, config_end = self._read_config_and_commands()
            lines = f.readlines()
            for cs in self.commands:
                command_sign = self.commands[cs]
                # command_sign (eg. @1 or @2 ) are the initialized value of commands dict keys
                # which provided by the _read_commands function.
                data_section = False
                data = []
                for line in lines[config_end + 1:]:
                    if line.rfind(command_sign) > -1:
                        data_section = True
                        continue
                    if data_section and line != '\n':
                        d = _read_data(line)
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

        with open(output_path, 'w') as f:
            f.writelines(self.config)
            _write_from_dict(f, all_data)

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
            self.config["config"] = lines[:config_end + 1]
        return commands, config_end


class Hoc:

    def __init__(self):
        pass


class Amira_utils:

    def __init__(self):
        pass


def _read_data(line):
    print line
    print "hi"
    matches = re.findall('-?\d+\.\d+[e]?[+-]?\d+|\-?\d+[e]?', line)
    print matches
    if not matches:
        matches = [0.0]
    data = map(float, matches)
    return data


def _write_from_dict(data_file, data_dict):
    pass
