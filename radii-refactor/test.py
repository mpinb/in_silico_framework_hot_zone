"""
This file is used for testing IO.py.
"""
import os
import IO
from definitions import ROOT_DIR


def compare_points(p1, p2):
    if p1 == p2:
        return True
    else:
        return False


def __test_am_read():
    am_path = os.path.join(ROOT_DIR, 'test_files/S13_final_done_Alison_zScale_40.am')
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


__test_am_read()

__test_am_write()