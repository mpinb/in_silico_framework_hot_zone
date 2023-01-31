import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import dask
import time
from .helper_methods import write_video_from_images, write_gif_from_images, display_animation_from_images
import warnings
from barrel_cortex import inhibitory
warnings.filterwarnings("default", category=ImportWarning, module=__name__)  # let ImportWarnings show up when importing this module through Interface
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    warnings.warn("tqdm not found in the current environment. No progressbars will be displayed", ImportWarning)
    HAS_TQDM = False
from ipywidgets import interactive, VBox, HBox, widgets, Layout
try:
    import plotly.offline as py
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    warnings.warn("Plotly could not be imported. Interactive visualisations will not work.", ImportWarning)
import pandas as pd


class CellMorphologyVisualizer:
    """
    This class initializes from a cell object and extracts relevant Cell data to a format that lends itself easier to plotting.
    It contains useful methods for either plotting a cell morphology, the voltage along its body and its synaptic inputs.
    This can be visualized in static images or as time series (videos, gif, animation, interactive window).
    Also, the relevant cell information can also be exported to .vtk format for further visualization or interaction.
    """
    def __init__(self, cell):
        """
        Given a Cell object, this class initializes an object that is easier to work with
        """
        # Colors in which the synaptic input are going to be shown
        self.population_to_color_dict = {'INT': 'black', 'L4ss': 'orange', 'L5st':'cyan', 'L5tt':'lime',
                     'L6CC':'yellow', 'VPM':'red', 'L23':'magenta', 'inactive':'white'}
        
        # Angles in which our 3D plots are shown
        self.azim = 0  # Azimuth of the projection in degrees.
        self.dist = 10 # Distance of the camera to the object.
        self.elev = 30 # Elevation of the camera above the equatorial plane in degrees.
        self.roll = 0
        # Max and min voltages colorcoded in the cell morphology
        self.vmin = -80 # mV
        self.vmax = 20 # mV
        # time offset w.r.t. simulation start. useful if '0 ms' is supposed to refer to stimulus time
        self.time_offset = 0
        self.dpi = 72
        # Time in the simulation during which a synapse activation is shown during the visualization
        self.time_show_syn_activ = 2 # ms
        
        # Cell object
        self.cell = cell
        
        """Gather the necessary information to plot the cell morphology. This info is always necessary to plot the cell.
        morphology: pandas dataframe with points.
        Each point contains the x, y and z coordinates, a diameter and the section index
        of the neuron to which that point belongs.
        Sections are defined as the regions of cell morphology
        between 2 branching points / end of a branch.
        """
        self.line_pairs = []  # initialised below
        self.morphology = self._get_morphology()
        self.points = self.morphology[["x", "y", "z"]]
        self.diameters = self.morphology["diameter"]
        self.section_indices = self.morphology["section"]
        
        # Time points of the simulation
        self.simulation_times = np.array(cell.tVec)
        # Time points we want to visualize
        self.t_start = self.simulation_times[0]
        self.dt = self.simulation_times[1] - self.t_start
        self.t_step = (len(self.simulation_times)//10) * self.dt
        self.t_end = self.simulation_times[-1] - self.simulation_times[-1] % self.t_step
        self.times_to_show = np.empty(0)
        self._update_time(self.t_start, self.t_end, self.t_step)  # initialise time range to visualise
        
        """List contaning the voltage of the cell during a timeseries. Each element corresponds to a time point.
        Each element of the list contains n elements, being n the number of points of the cell morphology. 
        Hence, the value of each element is the voltage at each point of the cell morphology."""
        self.voltage_timeseries = []
        
        """List containing the synapse activations during a timeseries (Similarly to voltage_timeseries). 
        Each element corresponds to a time point. Each element is a dictionary where each key is the type of
        input population and the value is the list of active synapses for that type of population at that time point. 
        The list contains the 3d coordinates where each active synapse is located."""
        self.synapses_timeseries = []

    # "Private" methods (nothing is truly private in Python)
        
    def _get_morphology(self):
        '''
        Retrieve cell MORPHOLOGY from cell object.
        Fills the self.morphology attribute
        '''
        points = []
        for sec_n, sec in enumerate(self.cell.sections):
            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue
            # Point that belongs to the previous section (x, y, z and diameter)
            x = sec.parent.pts[-1][0]
            y = sec.parent.pts[-1][1]
            z = sec.parent.pts[-1][2]
            d = sec.parent.diamList[-1]
            points.append([x, y, z, d, sec_n]) 

            for i, pt in enumerate(sec.pts):
                # Points within the same section
                x = pt[0]
                y = pt[1]
                z = pt[2]
                self.line_pairs.append([i, i+1])
                d = sec.diamList[i]
                points.append([x, y, z, d, sec_n])
        morphology = pd.DataFrame(points, columns=['x','y','z','diameter','section'])
        return morphology
    
    def _plot_cell_voltage_synapses_in_morphology_3d(self, voltage, synapses, time_point, legends=True, save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
        are shown for a particular time point.
        Args:
            - voltage: voltage along the whole cell morphology for a particular time point
            - synapses: synapse activations for a particular timepoint
            - time_point: time of the simulation which we would like to visualize
            - Legends: whether the voltage and synapse activations legends should appear in the plot
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        # Plot morphology with colorcoded voltage
        cmap=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=plt.get_cmap('jet'))
        fig = plt.figure(figsize = (15,15), dpi = self.dpi)
        ax = plt.axes(projection='3d', proj_type = 'ortho')
        sections = np.unique(self.morphology['section'].values)
        for sec in sections:
            points = self.morphology.loc[self.morphology['section'] == sec]
            for i in points.index:
                if i!= points.index[-1]:
                    color = ((voltage[i]+voltage[i+1])/2 - self.vmin)/(self.vmax-self.vmin)
                    linewidth = (points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                    ax.plot3D([points.loc[i]['x'],points.loc[i+1]['x']],
                              [points.loc[i]['y'],points.loc[i+1]['y']],
                              [points.loc[i]['z'],points.loc[i+1]['z']], color=mpl.cm.jet(color) ,lw=linewidth)  

        # Plot synapse activations
        for population in synapses.keys():
            for synapse in synapses[population]:
                    color = self.population_to_color_dict[population]
                    ax.scatter3D(synapse[0],synapse[1],synapse[2],
                                 color=color, edgecolors='grey', s = 75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

        # Add legends
        if legends:
            for key in self.population_to_color_dict.keys():
                if key != 'inactive':
                    ax.scatter3D([],[],[],color=self.population_to_color_dict[key], label=key, edgecolor = 'grey', s = 75)
            ax.legend(frameon=False, bbox_to_anchor=(-0.1715, 0.65, 1., .1), fontsize=12, # -0.1715, 0.75, 1., .1
                      title='Time = {}ms\n\n'.format(np.around(time_point-self.time_offset,1)), title_fontsize=12)
            cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5]) # 0.64, 0.2, 0.05, 0.5
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=mpl.cm.jet),
                 ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)#, fraction=0.015, pad=-0.25)
            plt.axis('off')

        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.roll = self.roll
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

        if save != '':
            plt.savefig(save)#,bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
    
    def _get_voltages_at_timepoint(self, time_point):
        '''
        Retrieves the VOLTAGE along the whole cell morphology from cell object at a particular time point.
        Args:
         - time_point: time point from which we want to gather the voltage
        '''
        n_sim_point = np.argmin(np.abs(self.simulation_times - time_point))
        voltage_points = []
        for sec in self.cell.sections:
            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue
            
            # Add voltage at the last point of the previous section
            voltage_points.append(sec.parent.recVList[-1][n_sim_point])

            # Compute segments limits (voltage is measure at the middle of each segment, not section)
            segs_limits = [0]
            for j,seg in enumerate(sec):
                segs_limits.append(segs_limits[j]+(seg.x-segs_limits[j])*2)

            # Map the segments to section points to assign a voltage to each line connecting 2 points
            current_seg = 0
            next_seg_flag = False 
            for i in range(len(sec.pts)):
                if i != 0:
                    if next_seg_flag:
                        current_seg += 1
                        next_seg_flag = False 
                    if sec.relPts[i-1] < segs_limits[current_seg+1] < sec.relPts[i]:
                        if sec.relPts[i] - segs_limits[current_seg+1] > segs_limits[current_seg+1] - sec.relPts[i-1]:
                            current_seg += 1
                        else:
                            next_seg_flag = True     
                voltage_points.append(sec.recVList[current_seg][n_sim_point])
        return voltage_points

    def _get_soma_voltage_at_timepoint(self, time_point):
        '''
        Retrieves the VOLTAGE along the whole cell morphology from cell object at a particular time point.
        Args:
         - time_point: time point from which we want to gather the voltage
        '''
        n_sim_point = np.argmin(np.abs(self.simulation_times - time_point))
        sec = [sec for sec in self.cell.sections if sec.label == "Soma"][0] 
        return sec.recVList[0][n_sim_point]

    def _get_soma_voltage_between_timepoints(self, t_start, t_end, t_step):
        voltages=[]
        for t_ in np.arange(t_start, t_end+t_step, t_step):
            voltages.append(self._get_soma_voltage_at_timepoint(t_))
        return voltages

    def _get_timeseries_voltage(self):
        '''
        Retrieves VOLTAGE along the whole cell body during a set of time points (specified in self.times_to_show).
        Fills the self.voltage_timeseries attribute.
        Only does so when it has not been computed yet.
        '''
        if len(self.voltage_timeseries)!=0:
            return  # We have already retrieved the voltage timeseries
        
        t1 = time.time()
        for time_point in self.times_to_show: # For each frame of the video/animation
            voltage = self._get_voltages_at_timepoint(time_point)
            self.voltage_timeseries.append(voltage)
        t2 = time.time()
        print('Voltage retrieval runtime (s): ' + str(np.around(t2-t1,2)))
    
    def _get_synapses_at_timepoint(self, time_point):
        '''
        Retrieves the SYNAPSE ACTIVATIONS at a particular time point.
        Args:
         - time_point: time point from which we want to gather the synapse activations
        '''
        def match_model_celltype_to_PSTH_celltype(celltype):
            if '_' in celltype:
                celltype = celltype.split('_')[0]
            if celltype in inhibitory or celltype == 'INH':
                key = 'INT'
            elif celltype in ('L4ss', 'L4py', 'L4sp'):
                key = 'L4ss'
            elif celltype == 'L5st':
                key = 'L5st'
            elif celltype == 'L5tt':
                key = 'L5tt'
            elif celltype == 'L6cc':
                key = 'L6CC'
            elif celltype == 'VPM':
                key = 'VPM'
            elif celltype in ('L2','L34'):
                key = 'L23'
            elif celltype in ('L6ct', 'L6ccinv'):
                key = 'inactive'
            else:
                raise ValueError(celltype)   
            return key
        
        synapses = {'INT': [], 'L4ss': [], 'L5st':[], 'L5tt':[], 'L6CC':[], 'VPM':[], 'L23':[]}
        for population in self.cell.synapses.keys():
            for synapse in self.cell.synapses[population]:
                for spikeTime in synapse.preCell.spikeTimes:
                    if time_point-self.time_show_syn_activ < spikeTime < time_point+self.time_show_syn_activ:
                        population_name = match_model_celltype_to_PSTH_celltype(population)
                        synapses[population_name].append([synapse.coordinates[0],
                                                          synapse.coordinates[1],
                                                          synapse.coordinates[2]])
        return synapses
        
    def _get_timeseries_synapses(self): 
        '''
        Retrieves the SYNAPSE ACTIVATIONS during a set of time points (specified in self.time).
        Fills the self.synapses_timeseries attribute.
        '''
        if len(self.synapses_timeseries)!=0:
            return # We have already retrieved the synapses timeseries
        
        t1 = time.time()
        for time_point in self.times_to_show: # For each frame of the video/animation
            synapses = self._get_synapses_at_timepoint(time_point)
            self.synapses_timeseries.append(synapses)
        t2 = time.time()
        print('Synapses retrieval runtime (s): ' + str(np.around(t2-t1,2)))
            
    def _update_time(self, t_start=None, t_end=None, t_step=None):
        """Checks if the specified time range equals the previously defined one. If not, updates the time range.
        If all arguments are None, does nothing. Useful for defining default time range
        TODO: what if a newly defined t_step does not match the simulation dt?

        Args:
            t_start (float): start time
            t_end (float): end time
            t_step (float): time interval
        """
        if not all([e is None for e in (t_start, t_end, t_step)]):
            # At least one of the time range parameters needs to be updated
            if t_start is not None:
                assert t_start >= self.simulation_times[0], "Specified t_start is earlier than the simulation time of {} ms".format(self.simulation_times[0])
                self.t_start = t_start  # update start if necessary
            # check if closed interval is possible
            if t_end is not None:
                assert t_end <= self.simulation_times[-1], "Specified t_end exceeds the simulation time of {} ms".format(self.simulation_times[-1])
                self.t_end = t_end
            if t_step is not None:
                self.t_step = t_step
            
            new_time = np.arange(self.t_start, self.t_end + self.t_step, self.t_step)
            if len(self.times_to_show) != 0:  # there were previously defined times
                if len(self.times_to_show) == len(new_time):
                    if (self.times_to_show != new_time).any():  # there are timepoints that are newly defined
                        self.voltage_timeseries = []
                        self.synapses_timeseries = []
                else:  # there were no times previously defined
                    self.voltage_timeseries = []
                    self.synapses_timeseries = []
            self.times_to_show = new_time  
        
    def _timeseries_images_cell_voltage_synapses_in_morphology_3d(self, path, t_start=None, t_end=None, t_step=None, neuron_rotation = None, client=None):
        '''
        Creates a list of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. These images will then be used for a time-series visualization (video/gif/animation)
        and in each image the neuron rotates a bit (3 degrees) over its axis.

        The parameters :@param t_start:, :@param t_end: and :@param t_step: will define the self.time attribute

        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - path: path were the images should be stored
            - client: dask client for parallelization
        '''
        if os.path.exists(path):
            if os.listdir(path):
                print('Images already generated, they will not be generated again. Please, change the path name or delete the current one.')
                return
        else:
            os.mkdir(path)
        
        # Gathers the voltage and synapse activations time series. 
        # Then images are generated for each specified time step.
        self._update_time(t_start,t_end,t_step)
        self._get_timeseries_voltage()
        self._get_timeseries_synapses()
        
        out = []
        azim_ = self.azim
        count = 0
        
        # Create images for animation/video
        t1 = time.time()
        for voltage,synapses in zip(self.voltage_timeseries, self.synapses_timeseries):
            time_point = self.times_to_show[count]
            count += 1
            filename = path+'/{0:0=5d}.png'.format(count)
            out.append(plot_cell_voltage_synapses_in_morphology_3d(
                        morphology=self.morphology, voltage=voltage, synapses=synapses, 
                        time_point=time_point, save=filename, population_to_color_dict=self.population_to_color_dict,
                        azim=self.azim, dist=self.dist, roll = self.roll, elev=self.elev, vmin=self.vmin, vmax=self.vmax, legends=True,
                        time_offset = self.time_offset, dpi = self.dpi))
            self.azim += neuron_rotation
        self.azim = azim_
        futures = client.compute(out)
        client.gather(futures)
        t2 = time.time()
        print('Images generation runtime (s): ' + str(np.around(t2-t1,2)))
            
    def _write_vtk_frame(self, out_name, out_dir, time_point, scalar_data=None):
        '''
        Format in which a cell morphology (in a singular time point) (color-coded with scalar data, by default membrane voltage) 
        is saved to be visualized in paraview
        '''
        def header_(out_name_=out_name):
            h="# vtk DataFile Version 4.0\n{}\nASCII\nDATASET POLYDATA\n".format(out_name_)
            return h

        def points_str_(points_):
            p = ""
            for p_ in points_:
                line = ""
                for comp in p_:
                    line += str(round(comp, 3))
                    line += " "
                p += str(line[:-1])
                p += "\n"
            return p

        def line_pairs_str_(line_pairs_):
            """
            This method adds lines to the vtk mesh pair per pair, such that each line segment is always defined by only two points.
            This is useful for when you want to specify a varying diameter
            """
            n = 0
            l = ""
            for p1, p2 in line_pairs_:
                l += "2 {} {}\n".format(p1, p2)
                n += 1
            return l
        
        def diameter_str_(diameters_):
            diameter_string = ""
            for d in diameters_:
                diameter_string += str(d)
                diameter_string += "\n"
            return diameter_string
        
        def membrane_voltage_str_(data_):
            """
            Given an array :@param data: of something, returns a string where each row of the string is a single element of that array.
            """
            v_string = ""
            for d in data_:
                v += str(d)
                v += "\n"
            return v_string
        
        # write out all data to .vtk file
        with open(os.join(out_dir, out_name)+"_{:06d}.vtk".format(time_point), "w+", encoding="utf-8") as of:
            of.write(header_(out_name))

            # Points
            of.write("POINTS {} float\n".format(len(self.points)))
            of.write(points_str_(self.points))

            # Line
            of.write("LINES {} {}\n".format(len(self.line_pairs), 3*len(self.line_pairs)))  # LINES n_cells total_amount_integers
            # e.g. 2 16 765 means index 765 and 16 define a single line, the leading 2 defines the amount of points that define a line
            of.write(line_pairs_str_(self.line_pairs))

            # Diameters
            of.write("POINT_DATA {}\nSCALARS Diameter float 1\nLOOKUP_TABLE default\n".format(len(self.diameters)))  # SCALARS name_of_data data_type n_components
            of.write(diameter_str_(self.diameters))

            # Scalar data (as of now only membrane voltages)
            if scalar_data is not None:
                of.write("CELL_DATA {}\nSCALARS Vm float 1\nLOOKUP_TABLE default\n".format(len(self.morphology)))
                of.write(membrane_voltage_str_(scalar_data))
    
    # Public methods

    def show_morphology_3d(self, save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D
        Args:
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        fig = plt.figure(figsize = (15,15), dpi = self.dpi)
        ax = plt.axes(projection='3d', proj_type = 'ortho')
        sections = np.unique(self.morphology['section'].values)
        for sec in sections:
            points = self.morphology.loc[self.morphology['section'] == sec]
            for i in points.index:
                if i!= points.index[-1]:
                    linewidth = (points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                    ax.plot3D([points.loc[i]['x'],points.loc[i+1]['x']],
                              [points.loc[i]['y'],points.loc[i+1]['y']],
                              [points.loc[i]['z'],points.loc[i+1]['z']], color='grey' ,lw=linewidth)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

        if save != '':
            plt.savefig(save)#,bbox_inches='tight')
        if plot:
            plt.show()
        
    def show_voltage_in_morphology_3d(self, time_point, legends=True, save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage for a particular time point.
        Args:
            - Time_point: time of the simulation which we would like to visualize
            - Legends: whether the voltage legend should appear in the plot
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        voltage = self._get_voltages_at_timepoint(time_point)
        
        # Plot morphology with colorcoded voltage
        cmap=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=plt.get_cmap('jet'))
        fig = plt.figure(figsize = (15,15), dpi = self.dpi)
        ax = plt.axes(projection='3d', proj_type = 'ortho')
        sections = np.unique(self.morphology['section'].values)
        for sec in sections:
            points = self.morphology.loc[self.morphology['section'] == sec]
            for i in points.index:
                if i!= points.index[-1]:
                    color = ((voltage[i]+voltage[i+1])/2 - self.vmin)/(self.vmax-self.vmin)
                    linewidth = (points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                    ax.plot3D([points.loc[i]['x'],points.loc[i+1]['x']],
                              [points.loc[i]['y'],points.loc[i+1]['y']],
                              [points.loc[i]['z'],points.loc[i+1]['z']], color=mpl.cm.jet(color) ,lw=linewidth)  
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        
        # Add legends
        if legends:
            ax.legend(frameon=False, bbox_to_anchor=(-0.1715, 0.65, 1., .1), fontsize=12, # -0.1715, 0.75, 1., .1
                      title='Time = {}ms\n\n'.format(np.around(time_point,1)), title_fontsize=12)
            cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5]) # 0.64, 0.2, 0.05, 0.5
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=mpl.cm.jet),
                 ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)#, fraction=0.015, pad=-0.25)
            plt.axis('off')
        
        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.roll = self.roll
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

        if save != '':
            plt.savefig(save)#,bbox_inches='tight')
        if plot:
            plt.show()
    
    def show_voltage_synapses_in_morphology_3d(self, time_point, time_show_syn_activ=2, legends=True, save='',plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
        are shown for a particular time point.
        Args:
            - Time_point: time of the simulation which we would like to visualize
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - Legends: whether the voltage and synapse activations legends should appear in the plot
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        self.time_show_syn_activ = time_show_syn_activ
        voltage = self._get_voltages_at_timepoint(time_point)
        synapses = self._get_synapses_at_timepoint(time_point)
        self._plot_cell_voltage_synapses_in_morphology_3d(voltage,synapses,time_point,legends=legends,save=save,plot=plot, time_offset = self.time_offset)
      
    def write_gif_voltage_synapses_in_morphology_3d(self,images_path, out_path, client=None, t_start=None, t_end=None, t_step=None,
                                                   time_show_syn_activ=2, frame_duration=40):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a gif.

        Args:
            - images_path: path where the images for the gif will be generated
            - out_path: dir where the gif will be generated + name of the gif
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - client: dask client for parallelization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - frame_duration: duration of each frame in ms
            t_start, t_end and t_step will define the self.time attribute
        '''            
        if client is None:
            raise ValueError("Please provide a dask client object for the client argument")
        self._update_time(t_start, t_end, t_step)
        self.time_show_syn_activ = time_show_syn_activ
        self._timeseries_images_cell_voltage_synapses_in_morphology_3d(images_path, self.t_start, self.t_end, self.t_step, client)
        files = [os.path.join(images_path, f) for f in os.listdir(images_path)]
        files.sort()
        write_gif_from_images(out_path,files,frame_duration)
         
    def write_video_voltage_synapses_in_morphology_3d(self, images_path, out_path, t_start=None, t_end=None, t_step=None, client=None, time_show_syn_activ=2, framerate=12, quality=5, codec='mpeg4'):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a video.
        Args:
            - images_path: path where the images for the video will be generated
            - out_path: dir where the video will be generated + name of the video
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - client: dask client for parallelization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            t_start, t_end and t_step will define the self.time attribute
        '''
        if client is None:
            raise ValueError("Please provide a dask client object for the client argument")
        self._update_time(t_start, t_end, t_step)
        self.time_show_syn_activ = time_show_syn_activ
        self._timeseries_images_cell_voltage_synapses_in_morphology_3d(images_path, self.t_start, self.t_end, self.t_step, client)
        write_video_from_images(images_path, out_path, fps=framerate, quality=quality, codec=codec)
          
    def display_animation_voltage_synapses_in_morphology_3d(self, images_path, t_start=None, t_end=None, t_step=None, 
                                                            client=None, neuron_rotation = 3, time_show_syn_activ=2):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a python animation.
        Args:
            - images_path: path where the images for the gif will be generated
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - client: dask client for parallelization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            t_start, t_end and t_step will define the self.time attribute
        '''        
        if client is None:
            raise ValueError("Please provide a dask client object for the client argument")
        self._update_time(t_start, t_end, t_step)
        self.time_show_syn_activ = time_show_syn_activ
        self._timeseries_images_cell_voltage_synapses_in_morphology_3d(images_path, self.t_start, self.t_end, self.t_step, neuron_rotation, client)
        display_animation_from_images(images_path, 1, embedded=True)
        
    def display_interactive_voltage_in_morphology_3d(self, t_start=None, t_end=None, t_step=None, time_show_syn_activ=2, vmin=None, vmax=None, renderer="notebook_connected", color_map='jet', background_color="rgb(180,180,180)"):
        ''' 
        TODO: add synapse activations!
        TODO: add dendritic and somatic AP as secondary subplot
        Setup plotly for rendering in notebooks. Shows an interactive 3D render of the Cell with the following data overlayed:
        - Membrane voltage
        - Section ID
        - 3D Coordinates
        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - renderer
            - background_color: just some grey by default.
            t_start, t_end and t_step will define the self.time attribute
        Returns:
            ipywidgets.VBox object: an interactive render of the cell.
        '''
        self._update_time(t_start, t_end, t_step)
        self._get_timeseries_voltage()
        vmin = vmin if vmin is not None else self.vmin
        vmax = vmax if vmax is not None else self.vmax
        transparent="rgba(0, 0, 0, 0)"
        ax_layout = dict(
            backgroundcolor=transparent,
            gridcolor=transparent,
            showbackground=True,
            zerolinecolor=transparent,
            visible=False
        )
        
        py.init_notebook_mode()
        pio.renderers.default = renderer
        # Initialize a dataframe. This may seem inefficient, but plotly does this anyways whenever you pass data. 
        # Might as well explicitly do it yourself with more control
        df = self.morphology.copy()
        t_idx = np.where(self.times_to_show == self.t_start)[0][0]
        df['voltage'] = self.voltage_timeseries[t_idx]
        
        # Create figure
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", 
            hover_data=["x","y","z","section", "diameter","voltage"],
            color="voltage", range_color=[vmin, vmax],
            size="diameter", color_continuous_scale=color_map)
        fig.update_traces(marker = dict(line = dict(width = 0)))  # remove outline of markers
        fig.update_layout(scene=dict(
                                     xaxis = ax_layout,
                                     yaxis = ax_layout,
                                     zaxis = ax_layout,
                                     bgcolor=transparent  # turn off background for just the scene
                                        ),
                             plot_bgcolor=background_color,
                             paper_bgcolor=background_color,
                             coloraxis_colorbar=dict(title="V_m (mV)")
                             )

        fig_soma_voltage = px.line(
            x=np.arange(self.t_start, self.t_end+self.dt, self.dt), y=self._get_soma_voltage_between_timepoints(self.t_start, self.t_end, self.dt),
            labels={
                "x": "time (ms)",
                "y": "Membrane voltage (mV)"},
                 title="Soma membrane voltage"
                 )
        f_trace = go.FigureWidget(fig_soma_voltage.data, fig_soma_voltage.layout)

        # create FigureWidget from figure
        f = go.FigureWidget(data=fig.data, layout=fig.layout)

        # update color scale
        def _update(time_point):
            """
            Function that gets called whenever the slider gets changed. Requests the membrane voltages at a certain time point
            """
            round_floats = 2
            t_idx = np.where(self.times_to_show == time_point)[0][0]
            f.update_traces(marker={"color": np.round(self.voltage_timeseries[t_idx], round_floats)})
            f.layout.title = "Membrane voltage at time={} ms".format(time_point)
            f_trace.update_layout(
                shapes=[dict(
                    type= 'line',
                    yref= 'paper', y0= self.vmin, y1=self.vmax,
                    xref= 'x', x0=time_point, x1=time_point
                    )
                    ])
            return fig

        # display the FigureWidget and slider with center justification
        slider = interactive(
            _update, 
            time_point = widgets.FloatSlider(min=self.t_start, max=self.t_end, step=self.t_step, value=0, layout=Layout(width='800px'), background_color=background_color)
            )
        hb = HBox((f, f_trace))
        vb = VBox((hb, slider))
        vb.layout.align_items = 'center'
        return vb
               
    def write_vtk_frames(self, out_name="frame", out_dir=".", t_start=None, t_end=None, t_step=None):
        '''
        Format in which a cell morphology timeseries (color-coded with voltage) is saved to be visualized in paraview

        TODO: remove duplicate rows used to connect sections
        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - out_name: name of the file (not path, the file will be generated in out_dir)
            - out_dir: path where the images for the gif will be generated
        '''
        self._update_time(t_start, t_end, t_step)
        self._get_timeseries_voltage()
        
        progress = self.times_to_show
        if HAS_TQDM:
            progress = tqdm(t, desc="Writing vtk frames to {}".format(out_dir))
        for t in progress:
                self._write_vtk_frame(out_name, out_dir, scalar_data=self.voltage_timeseries[t])

@dask.delayed
def plot_cell_voltage_synapses_in_morphology_3d(morphology, voltage, synapses, time_point, save, population_to_color_dict,
                                                 azim=0, dist=10, elev=30, roll = 0, vmin=-75, vmax = -30, legends=True,
                                                 time_offset = 0, dpi = 72):
    '''
    Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
    are shown for a particular time point.
    Dask delayed function useful for parallelization of images generation. This dask delayed function cannot be part of the
    visualization class, dask does not allow it. TODO: find for a possible solution.
    Args:
        - morphology: pandas dataframe with points. Each point contains the x, y and z coordinates, a diameter and the section
          of the neuron to which that point belongs. Each section of a neuron is limited by a branching point or the end of 
          the neuron morphology.
        - voltage: voltage of the neuron along its morphology for a particular time point. Each element corresponds to the 
          voltage at each point of the neuron morphology (indexes match to the ones in the morphology dataframe).
        - synapses: active synapses for a particular time point. Dictionary where each key is the type of population
          and the value a list of the coordinates where each synapse of that population is located.
        - time_point: time point of the simulation which we want to visualize
        - save: path where the plot will be saved
        - population_to_color_dict: dictionary that indicates in which color each input population synapse will be shown.
        - azim: int or float: Azimuth of the projection in degrees. Default: 0
        - dist: int or float: distance of the camera to the object. Default: 10
        - elev: int or float: elevation of the camera above the equatorial plane in degrees. Default: 30
        - vmin: min voltages colorcoded in the cell morphology
        - vmax: max voltages colorcoded in the cell morphology
        - legends: whether the voltage and synapse activations legends should appear in the plot
    '''
    # Plot morphology with colorcoded voltage
    cmap=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=plt.get_cmap('jet'))
    fig = plt.figure(figsize = (15,15), dpi = dpi)
    ax = plt.axes(projection='3d', proj_type = 'ortho')
    sections = np.unique(morphology['section'].values)
    for sec in sections:
        points = morphology.loc[morphology['section'] == sec]
        for i in points.index:
            if i!= points.index[-1]:
                color = ((voltage[i]+voltage[i+1])/2 - vmin)/(vmax-vmin)
                linewidth = (points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                ax.plot3D([points.loc[i]['x'],points.loc[i+1]['x']],
                          [points.loc[i]['y'],points.loc[i+1]['y']],
                          [points.loc[i]['z'],points.loc[i+1]['z']], color=mpl.cm.jet(color) ,lw=linewidth) 

    # Plot synapse activations
    for population in synapses.keys():
        for synapse in synapses[population]:
                color = population_to_color_dict[population]
                ax.scatter3D(synapse[0],synapse[1],synapse[2], color=color, edgecolors='grey', s = 75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    
    # Add legends
    if legends:
        for key in population_to_color_dict.keys():
            if key != 'inactive':
                ax.scatter3D([],[],[],color=population_to_color_dict[key], label=key, edgecolor = 'grey', s = 75)
        ax.legend(frameon=False, bbox_to_anchor=(-0.1715, 0.65, 1., .1), fontsize=12, # -0.1715, 0.75, 1., .1
                  title='Time = {}ms\n\n'.format(np.around(time_point-time_offset,1)), title_fontsize=12)
        cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5]) # 0.64, 0.2, 0.05, 0.5
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=mpl.cm.jet),
             ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)#, fraction=0.015, pad=-0.25)
        plt.axis('off')
        
    ax.azim = azim
    ax.dist = dist
    ax.elev = elev
    ax.roll = roll
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])
    
    plt.savefig(save)#,bbox_inches='tight')
    plt.close()
