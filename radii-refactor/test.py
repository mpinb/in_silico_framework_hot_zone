"""
This file is used for testing IO.py.
"""
import os
import IO
from definitions import ROOT_DIR


def __test_am_read():
    am_path = os.path.join(ROOT_DIR, 'test_files/S13_final_done_Alison_zScale_40.am')
    print "------"

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
    print 'all_data["EDGE { int EdgeLabels } "]'
    print am_object.all_data["EDGE { int EdgeLabels } "]
    print "-------"
    return 0


__test_am_read()