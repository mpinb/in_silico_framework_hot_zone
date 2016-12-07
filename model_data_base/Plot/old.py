import plotfunctions
import analyze

class Plot():
    def __init__(self, mdb):
        self.mdb = mdb
#         self.spaciotemporal_binning_cell = None
#         self.average_std = None
#         self.average_std_voltage = None
#         self.spike_histogram = None
        
    def spaciotemporal_binning_synapses(self, \
                                        soma_distance_bins = 50, \
                                        min_time = 0, \
                                        max_time = 300, \
                                        time_distance_bins = 1):
        
        pixels = analyze.spaciotemporal_binning.universal(
            self.mdb['synapse_activation'],\
            'soma_distance', \
            spacial_distance_bins = soma_distance_bins, \
            min_time = min_time, \
            max_time = max_time, \
            time_distance_bins = time_distance_bins)
    
        return plotfunctions.pixels2figure(pixels), pixels
    
    def spaciotemporal_binning_universal(self, df, distance_column,\
                                        spacial_distance_bins = 50, \
                                        min_time = 0, \
                                        max_time = 300, \
                                        time_distance_bins = 1):
        
        pixels = analyze.spaciotemporal_binning.universal(
            df,\
            distance_column, \
            spacial_distance_bins = spacial_distance_bins, \
            min_time = min_time, \
            max_time = max_time, \
            time_distance_bins = time_distance_bins)
    
        return plotfunctions.pixels2figure(pixels), pixels     
    
    def average_std_universal(self, groupby_attribute = None):
        pass
    
    def spike_raster_plot(self, groupby_attribute = None):
        pass
    
    def manytraces_voltage(self, groupby_attribute = None):
        pass
    
#     def spike_raster_plot_universal(self):
#         pass
    
    def PSTH(self):
        pass

        
class Analyze():
    def __init__(self,mdb):
        self.mdb = mdb
        self.spike_detection