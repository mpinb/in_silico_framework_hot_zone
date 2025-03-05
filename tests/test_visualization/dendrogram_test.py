from visualize.dendrogram import Dendrogram
import matplotlib.pyplot as plt
from tests import setup_synapse_activation_experiment

class TestDendrogram:
    def setup_class(self):
        self.cell = setup_synapse_activation_experiment()
    
    def test_dendrogram(self):
        d = Dendrogram(self.cell)
        fig = d.plot()
        plt.show()