"""
This file is used for testing IO.py.
"""
import os

import numpy as np

import IO
import thickness as th
import pipeline
from definitions import ROOT_DIR
import pandas as pd


def compare_points(p1, p2):
    if p1 == p2:
        return True
    else:
        return False


def __test_am_read():
    am_path = os.path.join(ROOT_DIR, 'test_files/S13_final_done_Alison_zScale_40.am')
    print am_path
    print "***********"
    print "TEST Am.read() method:"
    print "***********"

    #   Test 1
    print "TEST 1"
    print "print the test file path:"
    print am_path
    print "------"
    am_object = IO.Am(am_path)

    #   Test 2
    print "TEST 2"
    print "am_object:"
    print "output_path: " + am_object.output_path
    print "input_path: " + am_object.input_path
    print "-------"

    #  Test 3
    print "TEST 3"
    am_object.read()
    print "commands: "
    print am_object.commands
    print "-------"

    #   Test 4
    print "TEST 4"
    print 'profile_data["POINT { float[3] EdgePointCoordinates }"]'
    defined_point = [1.849200057983398E01, 5.106000137329102E01, 1.310999989509583E00]
    point = am_object.all_data["POINT { float[3] EdgePointCoordinates }"][3]

    print "The point read from the file is as the same as the one from the " \
          "Am.read() method: " + \
          str(compare_points(defined_point, point))
    print "-------"

    del am_object


def __test_am_write():
    print "***********"
    print "TEST Am.write() method:"
    print "***********"
    am_path = os.path.join(ROOT_DIR, 'test_files/S13_final_done_Alison_zScale_40.am')
    am_object = IO.Am(am_path)
    am_object.read()

    #   Test 1
    am_object.write()


def __test_correct_seed():
    print "***********"
    print "TEST thicknesses._correct_seed() method:"
    print "***********"
    # image point and its value, extracted using ImageJ:
    # x = 2400, y = 2364, value = 150
    # the maximum value in a area of thickness 10 micron is 181 at [2403, 2447]
    image_point = [2400, 2364]
    image_file = os.path.join(ROOT_DIR, 'test_files/S13_max_z_projection.tif')
    rx_object = th.ThicknessExtractor([], image_file)
    corrected_point = rx_object._correct_seed(image_point)
    print "The _correct_seed function correct the point [2400, 2364] to  [2403, 2447]:"
    if corrected_point == [2403, 2447]:
        print "TRUE"
    else:
        print "FALSE"

    a = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    a = np.array(a)
    np.testing.assert_array_almost_equal(th._crop_image(a, (0, 0), 0), np.array([[1]]))
    np.testing.assert_array_almost_equal(th._crop_image(a, (0, 0), 1), np.array([[0, 0, 0], [0, 1, 2], [0, 2, 3]]))
    np.testing.assert_array_almost_equal(th._crop_image(a, (0, 4), 1), np.array([[0, 0, 0], [4, 5, 0], [5, 6, 0]]))
    np.testing.assert_array_almost_equal(th._crop_image(a, [1, 2], 1, circle=True),
                                         np.array([[0, 3, 0], [3, 4, 5], [0, 5, 0]]))


def __test_pipeline():
    am_folder_path = os.path.join(ROOT_DIR, 'test_files/am_files')
    tif_folder_path = os.path.join(ROOT_DIR, 'test_files/image_files')
    hoc_file_path = os.path.join(ROOT_DIR, 'test_files/WR58_Cell5_L5TT_Final.hoc')
    output_folder_path = os.path.join(ROOT_DIR, 'test_files/output')
    bijective_points_path = os.path.join(ROOT_DIR, 'test_files/manual_landmarks.landmarkAscii')

    p = pipeline.ExtractThicknessPipeline()

    p.set_am_paths_by_folder(am_folder_path)
    print "set_am_paths_by_folder test, should print the first am file"
    print p.am_paths[0]

    p.set_tif_paths_by_folder(tif_folder_path)
    print "set_tif_paths_by_folder test, should print the first tif file"
    print p.tif_paths[0]
    print os.path.basename(p.am_paths[0])

    p.set_output_path(output_folder_path)
    print "set_output_path test, should print the output folder path"
    print p.set_output_path

    p.set_hoc_file(hoc_file_path)
    print "set_hoc_file test, should print the hoc file path"
    print p.hoc_file

    p.set_thickness_extractor_parameters()
    p.set_bijective_points_by_file(bijective_points_path)

#   p.set_client_for_parallelization("localhost", 8780)

    df = p.run()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

print "----------"
print "TEST IO.py"
print "----------"
# __test_am_read()
# __test_am_write()

print "----------"
print "TEST thicknesses.py"
print "----------"
# __test_correct_seed()

print "----------"
print "TEST pipeline.py"
print "----------"
__test_pipeline()
