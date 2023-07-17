from visualize.cell_morphology_visualizer import CellMorphologyVisualizer
import single_cell_parser as scp
import neuron
h = neuron.h
import os, unittest
from .context import *
from getting_started import getting_started_dir # path to getting started folder
from single_cell_parser.serialize_cell import *
from model_data_base.utils import silence_stdout
from tests import setup_current_injection_experiment

class TestCellMorphologyVisualizer(unittest.TestCase):
    def setUp(self):
        self.cell = setup_current_injection_experiment()
        self.cv = None

    def test_init(self):
        self.cv = CellMorphologyVisualizer(self.cell)
        self.assertIsInstance(self.cv, CellMorphologyVisualizer)

    # TODO: add more tests
