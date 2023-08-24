"""
This file is used for testing IO.py.
"""
import os

import numpy as np

from dendrite_thickness.thickness import IO, pipeline, utils
from dendrite_thickness.thickness import thickness as th
from .context import CURRENT_DIR
import pandas as pd

import logging
log = logging.getLogger(__name__)

AM_FILE = os.path.join(CURRENT_DIR, 'test_files', 'am_files', 'rest', 'S13_final_done_Alison_zScale_40.am')
IMAGE_FILE = os.path.join(CURRENT_DIR, 'test_files', 'image_files', 'rest', 'S13_max_z_projection.tif')


def test_am_read():
    log.info(AM_FILE)
    log.info("***********")
    log.info("TEST Am.read() method:")
    log.info("***********")

    #   Test 1
    log.info("TEST 1")
    log.info("print the test file path:")
    log.info(AM_FILE)
    log.info("------")
    am_object = IO.Am(AM_FILE)

    #   Test 2
    log.info("TEST 2")
    log.info("am_object:")
    log.info("output_path: " + am_object.output_path)
    log.info("input_path: " + am_object.input_path)
    log.info("-------")

    #  Test 3
    log.info("TEST 3")
    am_object.read()
    log.info("commands: ")
    log.info(am_object.commands)
    log.info("-------")

    #   Test 4
    log.info("TEST 4")
    log.info('profile_data["POINT { float[3] EdgePointCoordinates }"]')
    defined_point = [1.849200057983398E01, 5.106000137329102E01, 1.310999989509583E00]
    point = am_object.all_data["POINT { float[3] EdgePointCoordinates }"][3]

    log.info("The point read from the file is as the same as the one from the " \
          "Am.read() method: " + \
          str(utils.are_same_points(defined_point, point)))
    log.info("-------")

    del am_object


def test_am_write():
    log.info("***********")
    log.info("TEST Am.write() method:")
    log.info("***********")
    am_object = IO.Am(AM_FILE)
    am_object.read()

    #   Test 1
    am_object.write()


def test_correct_seed():
    log.info("***********")
    log.info("TEST thicknesses._correct_seed() method:")
    log.info("***********")
    # image point and its value, extracted using ImageJ:
    # x = 2400, y = 2364, value = 150
    # the maximum value in a area of thickness 10 micron is 181 at [2403, 2447]
    image_point = [2400, 2364, 150]  # [x, y, value]
    rx_object = th.ThicknessExtractor([], image_file=IMAGE_FILE)
    corrected_point = rx_object._correct_seed(image_point)
    log.info("The _correct_seed function correct the point [2400, 2364] to  [2403, 2447]:")
    if corrected_point == [2403, 2447]:
        log.info("TRUE")
    else:
        log.info("FALSE")

    a = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]
    a = np.array(a)
    np.testing.assert_array_almost_equal(th._crop_image(a, (0, 0), 0), np.array([[1]]))
    np.testing.assert_array_almost_equal(th._crop_image(a, (0, 0), 1), np.array([[0, 0, 0], [0, 1, 2], [0, 2, 3]]))
    np.testing.assert_array_almost_equal(th._crop_image(a, (0, 4), 1), np.array([[0, 0, 0], [4, 5, 0], [5, 6, 0]]))
    np.testing.assert_array_almost_equal(th._crop_image(a, [1, 2], 1, circle=True),
                                         np.array([[0, 3, 0], [3, 4, 5], [0, 5, 0]]))


def test_pipeline():
    am_folder_path = os.path.join(CURRENT_DIR, 'test_files/am_files')
    tif_folder_path = os.path.join(CURRENT_DIR, 'test_files/image_files')
    hoc_file_path = os.path.join(CURRENT_DIR, 'test_files/WR58_Cell5_L5TT_Final.hoc')
    output_folder_path = os.path.join(CURRENT_DIR, 'test_files/output')
    bijective_points_path = os.path.join(CURRENT_DIR, 'test_files/manual_landmarks.landmarkAscii')

    p = pipeline.ExtractThicknessPipeline()

    p.set_am_paths_by_folder(am_folder_path)
    log.info("set_am_paths_by_folder test, should log.info the first am file")
    log.info(p.am_paths[0])

    p.set_tif_paths_by_folder(tif_folder_path)
    log.info("set_tif_paths_by_folder test, should log.info the first tif file")
    log.info(p.tif_paths[0])
    log.info(os.path.basename(p.am_paths[0]))

    p.set_output_path(output_folder_path)
    log.info("set_output_path test, should log.info the output folder path")
    log.info(p.set_output_path)

    p.set_hoc_file(hoc_file_path)
    log.info("set_hoc_file test, should log.info the hoc file path")
    log.info(p.hoc_file)

    # p.set_thickness_extractor_parameters()
    p.set_am_to_hoc_transformation_by_landmarkAscii(bijective_points_path)

#   p.set_client_for_parallelization("localhost", 8780)

    df = p.run()
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        log.info(df)
