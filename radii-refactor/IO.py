"""
IO Module

==========
This module contains methods to read and write data with different formats.


File extensions
----------------------

1. .am
Capabilities:
- Reading data points in 3d and their corresponding radius
- Writing data points in 3d and their corresponding radius
Limitations:
- Not able to read other data than 3d positional points and their radius
- Only able to read and write in ascii format (In this module in the class amira_utils the are methods one can use to
 convert amira binary format to ascii format, for more details look at the amira_utils class docstring)

2 .hoc
Capabilities:
- Reading data points in 3d and their corresponding radius
- Writing data points in 3d and their corresponding radius
Limitations:
- Not able to construct the tree structure from the hoc file.
- Not able to read the white matter data.

3 .hx
Capabilities:
- Read and Write data with the help of Amira.
Limitations:
- Need to connect to Amira.

Tests
-----

- The test functions are inside the test.py. One can also use them as example of how to use the functions.

"""
import re
import os
from random import randrange


class am:

    def __init__(self, input_path, output_path=None, commands=None):

        if output_path is None:
            output_path = os.path.dirname(input_path) + "output_" + str(randrange(100))
        if commands is None:
            commands = {'EdgePointCoordinates': 'POINT { float[3] EdgePointCoordinates }',
                        'thickness': 'POINT { float thickness }'}
        self.input_path = input_path
        self.output_path = output_path
        self.commands = commands
        self.points = []

    def read(self):
        """
        Reading data form am file in the order of provided am commands in the
        dictionary's command, the default dictionary contains the 3d positions data
        and 1d radius data.

        """
        commands_sign = []
        data = []
        all_data =[]
        config_end = 0
        with open(self.input_path, 'r') as f:
            lines = f.readlines()
            for c in self.commands:
                for l_number, line in enumerate(lines):
                    if line.rfind(c) > -1:
                        commands_sign.append("@" + line[line.rfind("@") + 1])
                        config_end = l_number
                        continue
            for cs in commands_sign:
                data_section = False
                for line in lines[config_end + 1:]:
                    if line.rfind(cs):
                        data_section = True
                        continue
                    if data_section and line != '\n':
                        d = _read_data(line)
                        data.append(d)
                    elif data_section and line == '\n':
                        data_section = False
                        all_data = [dt + [data[idt]] for idt, dt in enumerate(all_data)]
            self.points = all_data
        return all_data


class hoc:

    def __init__(self):
        pass


class amira_utils:

    def __init__(self):
        pass


def _read_data(line):
    matches = re.findall('-?\d+\.\d+[e]?[+-]?\d+', line)
    if not matches:
        matches = [0.0]
    data = map(float, matches)
    return data
