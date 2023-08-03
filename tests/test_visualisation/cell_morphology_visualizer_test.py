from visualize.cell_morphology_visualizer import CellMorphologyVisualizer
from .context import *
from single_cell_parser.serialize_cell import *
from tests import setup_current_injection_experiment
import six

class TestCellMorphologyVisualizer:
    def __init__(self):
        self.cell = setup_current_injection_experiment()
        self.cv = None

    def test_init(self):
        self.cv = CellMorphologyVisualizer(self.cell, align_trunk=six.PY3)  # don't align trunk in py2, ithas no scipy Rotation object
        self.assertIsInstance(self.cv, CellMorphologyVisualizer)

    # TODO: add more tests
