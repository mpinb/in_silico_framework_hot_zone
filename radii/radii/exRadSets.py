# This module extracts all of the radii sets, each set corresponds to an
# .am file, and writes the calculated radii in new .am files in a new folder.

# Inputs:
# path_to_am: path to the folder contains the initial .am files without radius.
# path_to_tif: path to the folder contains the tif images corresponds to .am
# files.
# path_to_output: path to the output folder and its name.

# Outputs:
# 1. calculats radii sets, and puts each set inside its corresponding .am file
# inside the given output folder path.

import os
import re
import radii as radi
import SimpleITK as sitk


def exRadSets(path_to_am, path_to_tif, path_to_output_folder):
    for spacialGraphFile in os.listdir(path_to_am):
        if spacialGraphFile.endswith(".am"):
            spacialGraphIndicator = re.findall(r'[sS]\d+', spacialGraphFile)[0]
            for imageFile in os.listdir(path_to_tif):
                if imageFile.startswith(spacialGraphIndicator):
                    #now the rest of the application will come here
                    #as simple as possible
