from visualize import CellVisualizer
import single_cell_parser as scp
import neuron
h = neuron.h
import os, unittest
from .context import *
from getting_started import getting_started_dir # path to getting started folder
from single_cell_parser.serialize_cell import *
from model_data_base.utils import silence_stdout
from .. import setup_current_injection_experiment

class Tests(unittest.TestCase):
    def setup_current_injection_experiment(self):
        self.cell = setup_current_injection_experiment()
        self.cv = None

    def test_init(self):
        self.cv = CellVisualizer(self.cell)
        self.assertIsInstance(self.cv, CellVisualizer)

    # TODO: add more tests