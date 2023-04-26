from scipy.spatial.transform import Rotation
from biophysics_fitting import get_main_bifurcation_section
import pandas as pd
from ipywidgets import interactive, VBox, HBox, widgets, Layout
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import dask
import time
from .helper_methods import write_video_from_images, write_gif_from_images, display_animation_from_images
import warnings
from barrel_cortex import inhibitory
# let ImportWarnings show up when importing this module through Interface
warnings.filterwarnings("default", category=ImportWarning, module=__name__)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    warnings.warn(
        "tqdm not found in the current environment. No progressbars will be displayed", ImportWarning)
    HAS_TQDM = False
try:
    import plotly.offline as py
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    warnings.warn(
        "Plotly could not be imported. Interactive visualisations will not work.", ImportWarning)


class CellMorphologyVisualizer:
    """
    This class initializes from a cell object and extracts relevant Cell data to a format that lends itself easier to plotting.
    It contains useful methods for either plotting a cell morphology, the voltage along its body and its synaptic inputs.
    This can be visualized in static images or as time series (videos, gif, animation, interactive window).
    Also, the relevant cell information can also be exported to .vtk format for further visualization or interaction.
    """

    def __init__(self, cell, align_trunk=True):
        """
        Given a Cell object, this class initializes an object that is easier to work with
        """
        # ---------------------------------------------------------------
        # Cell object
        self.cell = cell
        """The Cell object"""

        # ---------------------------------------------------------------
        # Visualization attributes
        self.population_to_color_dict = {'INT': 'black', 'L4ss': 'orange', 'L5st': 'cyan', 'L5tt': 'lime',
                                         'L6CC': 'yellow', 'VPM': 'red', 'L23': 'magenta', 'inactive': 'white'}
        """Colors in which the synaptic input are going to be shown"""

        ## Angles in which our 3D plots are shown
        self.azim = 0
        """Azimuth of the projection in degrees."""
        self.dist = 10
        """Distance of the camera to the object."""
        self.elev = 30
        """Elevation of the camera above the equatorial plane in degrees."""
        self.roll = 0
        self.neuron_rotation = 3
        """Rotation degrees of the neuron at each frame during timeseries visualization (in azimuth)"""
        self.dpi = 72
        """Image quality"""

        # ---------------------------------------------------------------
        # Morphology attributes
        ## Gather the necessary information to plot the cell morphology.
        ## This info is always necessary to plot the cell.
        
        self.line_pairs = []  # initialised below
        """Pairs of point indices that define a line, i.e. some cell segment"""
        self.morphology = self.__get_morphology()  # a pandas DataFrame
        """A pd.DataFrame containing point information, diameter and section ID"""
        if align_trunk:
            self.__align_trunk_with_z_axis()
        self.points = self.morphology[["x", "y", "z"]]
        """The 3D point coordinates of the cell"""
        self.diameters = self.morphology["diameter"]
        """Diameters of the cell segments"""
        self.section_indices = self.morphology["section"]
        """Section indices of the Cell segments"""

        # ---------------------------------------------------------------
        # Simulation-related parameters
        ## These only get initialised when the cell object actually contains simulation data.
        ## This info is not necessary to plot the cell morphology, but some more advanced methods need this information
        
        self.vmin = None  # mV
        """Max voltage colorcoded in the cell morphology (mV)"""
        self.vmax = None  # mV
        """Min voltage colorcoded in the cell morphology (mV)"""
        self.simulation_times = None
        """Time points of the simulation"""
        self.time_offset = None
        """Time offset w.r.t. simulation start. useful if '0 ms' is supposed to refer to stimulus time"""
        self.t_start = None
        """Time point where we want to start visualising.
        By default, this gets initialised to the start of the simulation."""
        self.t_end = None
        """Time point where the visualisation of the simulation stops.
        By default, this gets initialised to the end of the simulation."""
        self.dt = None
        """Time interval of the simulation"""
        # TODO: add support for variable timestep
        self.t_step = None
        """Time interval for visualisation. Does not have to equal the simulation time interval.
        By default, the simulation is chopped to the specified t_begin and t_end, and evenly divided in 10 timesteps."""
        self.times_to_show = None
        """An array of time points to visualize. Gets calculated from :param:self.t_begin, :param:self.t_end and :param:self.t_step"""

        self.scalar_data = None
        """Scalar data to overlay on the meuron morphology. If there is simulation data available, this is initialized as the membrane voltage, but ion currents are also possible"""
        self.possible_scalars = {'K_Pst.ik', 'Ih.m', 'Ca_LVAst.ik', 'Nap_Et2.ina', 'SK_E2.ik', 'Nap_Et2.h', 'K_Pst.h', 'Nap_Et2.m', 'NaTa_t.h', 'SK_E2.z', 'Ca_HVA.h', 'Ih.ihcn', 'K_Tst.ik',
                                 'Ca_HVA.m', 'Im.ik', 'NaTa_t.ina', 'NaTa_t.m', 'SKv3_1.m', 'Ca_LVAst.h', 'K_Pst.m', 'SKv3_1.ik', 'Ca_LVAst.m', 'Ca_HVA.ica', 'Ca_LVAst.ica', 'K_Tst.m',
                                 'CaDynamics_E2.cai', 'K_Tst.h', 'cai'}
        """Accepted keywords for scalar data other than membrane voltage."""

        self.voltage_timeseries = None
        """List contaning the voltage of the cell during a timeseries. Each element corresponds to a time point.
        Each element of the list contains n elements, being n the number of points of the cell morphology. 
        Hence, the value of each element is the voltage at each point of the cell morphology.
        None means it has no simulation data. Empty list means it has simulation data that has not been initialised yet."""

        self.synapses_timeseries = None
        """List containing the synapse activations during a timeseries (Similarly to voltage_timeseries). 
        Each element corresponds to a time point. Each element is a dictionary where each key is the type of
        input population and the value is the list of active synapses for that type of population at that time point. 
        The list contains the 3d coordinates where each active synapse is located.
        None means it has no simulation data. Empty list means it has simulation data that has not been initialised yet."""

        self.ion_dynamics_timeseries = None
        """List containing the ion dynamics during a timeseries (Similarly to voltage_timeseries). 
        Each element is a list corresponding to 1 timepoint, containing per-point info on the ion channel state or ion concentration.
        None means it has no simulation data. Empty list means it has simulation data that has not been initialised yet.
        """

        self.time_show_syn_activ = 2  # ms
        """Time in the simulation during which a synapse activation is shown during the visualization"""

        if self.__has_simulation_data():
            self.__init_simulation_data()

    # "Private" methods
    def __has_simulation_data(self):
        return len(self.cell.soma.recVList[0]) > 0

    def __assert_has_simulation_data(self):
        assert self.__has_simulation_data(), "This cell object has no simulation data, yet you are trying to plot data that is only available after a simulation. You can only plot its morphology, not its membrane voltage or synapses."

    def __init_simulation_data(self):
        # Max and min voltages colorcoded in the cell morphology
        self.vmin = -80  # mV
        self.vmax = 20  # mV

        """Time"""
        # Time points of the simulation
        self.simulation_times = np.array(self.cell.tVec)
        # Time offset w.r.t. simulation start. useful if '0 ms' is supposed to refer to stimulus time
        self.time_offset = 0
        # Time points we want to visualize (values by default)
        self.t_start = self.simulation_times[0]
        self.dt = self.simulation_times[1] - self.t_start
        # TODO: add support for variable timestep
        self.t_step = (len(self.simulation_times)//10) * self.dt
        self.t_end = self.simulation_times[-1] - \
            self.simulation_times[-1] % self.t_step
        self.times_to_show = np.empty(0)
        # initialise time range to visualise
        self.__update_times_to_show(self.t_start, self.t_end, self.t_step)

        """List contaning the voltage of the cell during a timeseries. Each element corresponds to a time point.
        Each element of the list contains n elements, being n the number of points of the cell morphology. 
        Hence, the value of each element is the voltage at each point of the cell morphology."""
        self.voltage_timeseries = []

        """List containing the synapse activations during a timeseries (Similarly to voltage_timeseries). 
        Each element corresponds to a time point. Each element is a dictionary where each key is the type of
        input population and the value is the list of active synapses for that type of population at that time point. 
        The list contains the 3d coordinates where each active synapse is located."""
        self.synapses_timeseries = []

        # Time in the simulation during which a synapse activation is shown during the visualization
        self.time_show_syn_activ = 2  # ms
        self.scalar_data = {"voltage": self.voltage_timeseries}

    def __align_trunk_with_z_axis(self):
        """
        Calculates the polar angle between the trunk and z-axis (zenith).
        Anchors the soma to (0, 0, 0) and aligns the trunk to the z-axis.

        Args:

        Returns:
        Nothing
        """

        assert len(self.morphology) > 0, "No morphology initialised yet"
        soma = [section for section in self.cell.sections if section.label == "Soma"][0]
        soma_center = np.mean(soma.pts, axis=0)
        # the bifurcation sections is the entire section: can be quite large
        # take last point, i.e. furthest from soma
        bifurcation = get_main_bifurcation_section(self.cell).pts[-1]
        soma_bif_vector = bifurcation - soma_center
        soma_bif_vector /= np.linalg.norm(soma_bif_vector)
        # angle with z-axis
        zenith = np.arccos(
            np.dot([0, 0, 1], soma_bif_vector)
        )
        xy_proj = [soma_bif_vector[0], soma_bif_vector[1], 0]
        xy_proj /= np.linalg.norm(xy_proj)
        # create vector to rotate about
        xy_proj_orth = [xy_proj[1], -xy_proj[0], 0]
        # rotation towards z-axis as rotation vector
        # as rotation vector: direction is axis to rotate about, norm is angle of rotation
        rot_vec = [e * zenith for e in xy_proj_orth]
        rotation = Rotation.from_rotvec(rot_vec)

        # Anchor soma to (0, 0, 0) and rotate trunk to align with z-axis
        self.morphology[['x', 'y', 'z']] = rotation.apply(
            [e - soma_center for e in self.morphology[['x', 'y', 'z']].values]
        )

    def __get_morphology(self):
        '''
        Retrieve cell MORPHOLOGY from cell object.
        Fills the self.morphology attribute
        '''
        t1 = time.time()

        points = []
        for sec_n, sec in enumerate(self.cell.sections):
            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue
            # Point that belongs to the previous section (x, y, z and diameter)
            x, y, z = sec.parent.pts[-1]
            d = sec.parent.diamList[-1]
            points.append([x, y, z, d, sec_n])

            for i, pt in enumerate(sec.pts):
                # Points within the same section
                x, y, z = pt
                self.line_pairs.append([i, i+1])
                d = sec.diamList[i]
                points.append([x, y, z, d, sec_n])
        morphology = pd.DataFrame(
            points, columns=['x', 'y', 'z', 'diameter', 'section'])
        t2 = time.time()
        print('Initialised simulation data in {} seconds'.format(np.around(t2-t1, 2)))
        return morphology

    def __plot_cell_voltage_synapses_in_morphology_3d(self, voltage, synapses, time_point, voltage_legend=True, synapse_legend=True, save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
        are shown for a particular time point.

        Args:
            - voltage: voltage along the whole cell morphology for a particular time point
            - synapses: synapse activations for a particular timepoint
            - time_point: time of the simulation which we would like to visualize
            - voltage_legend: whether the voltage legend should appear in the plot
            - synapse_legend: whether the synapse activations legend should appear in the plot
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        # Plot morphology with colorcoded voltage
        cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
            vmin=self.vmin, vmax=self.vmax), cmap=plt.get_cmap('jet'))
        fig = plt.figure(figsize=(15, 15), dpi=self.dpi)
        ax = plt.axes(projection='3d', proj_type='ortho')
        sections = np.unique(self.morphology['section'].values)
        for sec in sections:
            points = self.morphology.loc[self.morphology['section'] == sec]
            for i in points.index:
                if i != points.index[-1]:
                    color = ((voltage[i]+voltage[i+1])/2 -
                             self.vmin)/(self.vmax-self.vmin)
                    linewidth = (
                        points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                    ax.plot3D([points.loc[i]['x'], points.loc[i+1]['x']],
                              [points.loc[i]['y'], points.loc[i+1]['y']],
                              [points.loc[i]['z'], points.loc[i+1]['z']], color=mpl.cm.jet(color), lw=linewidth)

        # Plot synapse activations
        for population in synapses.keys():
            for synapse in synapses[population]:
                color = self.population_to_color_dict[population]
                ax.scatter3D(synapse[0], synapse[1], synapse[2],
                             color=color, edgecolors='grey', s=75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

        # Add legends
        if synapse_legend:
            for key in self.population_to_color_dict.keys():
                if key != 'inactive':
                    ax.scatter3D([], [], [
                    ], color=self.population_to_color_dict[key], label=key, edgecolor='grey', s=75)
            ax.legend(frameon=False, bbox_to_anchor=(-0.1715, 0.65, 1., .1), fontsize=12,  # -0.1715, 0.75, 1., .1
                      title='Time = {}ms\n\n'.format(np.around(time_point-self.time_offset, 1)), title_fontsize=12)
        if voltage_legend:
            # 0.64, 0.2, 0.05, 0.5
            cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5])
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=mpl.cm.jet),
                         ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)  # , fraction=0.015, pad=-0.25)
            plt.axis('off')

        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.roll = self.roll
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

        if save != '':
            plt.savefig(save)  # ,bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()

    def __get_voltages_at_timepoint(self, time_point):
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
            for j, seg in enumerate(sec):
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

    def __get_ion_dynamic_at_timepoint(self, time_point, ion_keyword):
        '''
        Retrieves the ion dynamics along the whole cell morphology from cell object at a particular time point.

        Args:
         - time_point: time point from which we want to gather the voltage
        '''
        n_sim_point = np.argmin(np.abs(self.simulation_times - time_point))
        ion_points = []
        for sec in self.cell.sections:
            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue

            # Add voltage at the last point of the previous section
            if not sec.parent.recordVars[ion_keyword] == []:
                ion_points.append(
                    sec.parent.recordVars[ion_keyword][-1][n_sim_point])
            else:
                ion_points.append(None)

            # Compute segments limits (voltage is measure at the middle of each segment, not section)
            segs_limits = [0]
            for j, seg in enumerate(sec):
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
                if not sec.recordVars[ion_keyword] == []:
                    ion_points.append(
                        sec.recordVars[ion_keyword][current_seg][n_sim_point])
                else:
                    ion_points.append(None)
        return ion_points

    def __get_soma_voltage_at_timepoint(self, time_point):
        '''
        Retrieves the VOLTAGE along the whole cell morphology from cell object at a particular time point.

        Args:
         - time_point: time point from which we want to gather the voltage
        '''
        n_sim_point = np.argmin(np.abs(self.simulation_times - time_point))
        sec = [sec for sec in self.cell.sections if sec.label == "Soma"][0]
        return sec.recVList[0][n_sim_point]

    def __get_soma_voltage_between_timepoints(self, t_start, t_end, t_step):
        voltages = []
        for t_ in np.arange(t_start, t_end+t_step, t_step):
            voltages.append(self.__get_soma_voltage_at_timepoint(t_))
        return voltages

    def __calc_voltage_timeseries(self):
        '''
        Retrieves VOLTAGE along the whole cell body during a set of time points (specified in self.times_to_show).
        Fills the self.voltage_timeseries attribute.
        Only does so when it has not been computed yet.

        Returns:
            Nothing. Updates the self.timeseries_voltage attribute
        '''
        if len(self.voltage_timeseries) != 0:
            return  # We have already retrieved the voltage timeseries

        t1 = time.time()
        for time_point in self.times_to_show:  # For each frame of the video/animation
            voltage = self.__get_voltages_at_timepoint(time_point)
            self.voltage_timeseries.append(voltage)
        self.scalar_data["voltage"] = self.voltage_timeseries
        t2 = time.time()
        print('Voltage retrieval runtime (s): ' + str(np.around(t2-t1, 2)))

    def __calc_ion_dynamics_timeseries(self, ion_keyword):
        '''
        Retrieves ion dynamics info along the whole cell body during a set of time points (specified in self.times_to_show).
        Fills the self.ion_dynamics attribute.
        Only does so when it has not been computed yet.

        Returns:
            Nothing. Updates the self.timeseries_voltage attribute
        '''
        if ion_keyword in self.scalar_data.keys():
            return  # We have already retrieved the voltage timeseries
        else:
            self.scalar_data[ion_keyword] = []

        t1 = time.time()
        for time_point in self.times_to_show:  # For each frame of the video/animation
            ion_dynamics = self.__get_ion_dynamic_at_timepoint(
                time_point, ion_keyword)
            self.scalar_data[ion_keyword].append(ion_dynamics)
        t2 = time.time()
        print('Ion dynamics retrieval runtime (s): ' + str(np.around(t2-t1, 2)))

    def __get_synapses_at_timepoint(self, time_point):
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
            elif celltype in ('L2', 'L34'):
                key = 'L23'
            elif celltype in ('L6ct', 'L6ccinv'):
                key = 'inactive'
            else:
                raise ValueError(celltype)
            return key

        synapses = {'INT': [], 'L4ss': [], 'L5st': [],
                    'L5tt': [], 'L6CC': [], 'VPM': [], 'L23': []}
        for population in self.cell.synapses.keys():
            for synapse in self.cell.synapses[population]:
                for spikeTime in synapse.preCell.spikeTimes:
                    if time_point-self.time_show_syn_activ < spikeTime < time_point+self.time_show_syn_activ:
                        population_name = match_model_celltype_to_PSTH_celltype(
                            population)
                        synapses[population_name].append([synapse.coordinates[0],
                                                          synapse.coordinates[1],
                                                          synapse.coordinates[2]])
        return synapses

    def __calc_synapses_timeseries(self):
        '''
        Retrieves the SYNAPSE ACTIVATIONS during a set of time points (specified in self.time).
        Fills the self.synapses_timeseries attribute.
        '''
        if len(self.synapses_timeseries) != 0:
            return  # We have already retrieved the synapses timeseries

        t1 = time.time()
        for time_point in self.times_to_show:  # For each frame of the video/animation
            synapses = self.__get_synapses_at_timepoint(time_point)
            self.synapses_timeseries.append(synapses)
        t2 = time.time()
        print('Synapses retrieval runtime (s): ' + str(np.around(t2-t1, 2)))

    def __update_times_to_show(self, t_start=None, t_end=None, t_step=None):
        """Checks if the specified time range equals the previously defined one. If not, updates the time range.
        If all arguments are None, does nothing. Useful for defining default time range
        
        Todo:
            what if a newly defined t_step does not match the simulation dt?

        Args:
            t_start (float): start time
            t_end (float): end time
            t_step (float): time interval
        """
        if not all([e is None for e in (t_start, t_end, t_step)]):
            # At least one of the time range parameters needs to be updated
            if t_start is not None:
                assert t_start >= self.simulation_times[0], "Specified t_start is earlier than the simulation time of {} ms".format(
                    self.simulation_times[0])
                assert t_start <= self.simulation_times[-1], "Specified t_start is later than the simulation end time of {} ms".format(
                    self.simulation_times[-1])
                self.t_start = t_start  # update start if necessary
            # check if closed interval is possible
            if t_end is not None:
                assert t_end <= self.simulation_times[-1], "Specified t_end exceeds the simulation time of {} ms".format(
                    self.simulation_times[-1])
                self.t_end = t_end
            if t_step is not None:
                self.t_step = t_step

            new_time = np.arange(
                self.t_start, self.t_end + self.t_step, self.t_step)
            if len(self.times_to_show) != 0:  # there were previously defined times
                if len(self.times_to_show) == len(new_time):
                    # there are timepoints that are newly defined
                    if (self.times_to_show != new_time).any():
                        self.ion_dynamics_timeseries = []
                        self.voltage_timeseries = []
                        self.synapses_timeseries = []
                else:  # there were no times previously defined
                    self.ion_dynamics_timeseries = []
                    self.voltage_timeseries = []
                    self.synapses_timeseries = []
            self.times_to_show = new_time

    def __timeseries_images_cell_voltage_synapses_in_morphology_3d(self, path, client=None, voltage_legend=True, synapse_legend=True):
        '''
        Creates a list of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. These images will then be used for a time-series visualization (video/gif/animation)
        and in each image the neuron rotates a bit (3 degrees) over its axis.

        The parameters :param:self.t_begin, :param:self.t_end and :param:self.t_step will define the :param:self.time attribute

        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - path: path were the images should be stored
            - client: dask client for parallelization
            - voltage_legend: whether the voltage legend should appear in the plot
            - synapse_legend: whether the synapse activations legend should appear in the plot
        '''
        if os.path.exists(path):
            if os.listdir(path):
                print('Images already generated, they will not be generated again. Please, change the path name or delete the current one.')
                return
        else:
            os.mkdir(path)

        # Gathers the voltage and synapse activations time series.
        # Then images are generated for each specified time step.
        self.__calc_voltage_timeseries()
        self.__calc_synapses_timeseries()

        out = []
        azim_ = self.azim
        count = 0

        # Create images for animation/video
        t1 = time.time()
        for voltage, synapses in zip(self.voltage_timeseries, self.synapses_timeseries):
            time_point = self.times_to_show[count]
            count += 1
            filename = path+'/{0:0=5d}.png'.format(count)
            out.append(plot_cell_voltage_synapses_in_morphology_3d(
                morphology=self.morphology, voltage=voltage, synapses=synapses,
                time_point=time_point, save=filename, population_to_color_dict=self.population_to_color_dict,
                azim=self.azim, dist=self.dist, roll=self.roll, elev=self.elev, vmin=self.vmin, vmax=self.vmax,
                voltage_legend=voltage_legend, synapse_legend=synapse_legend, time_offset=self.time_offset, dpi=self.dpi))
            self.azim += self.neuron_rotation
        self.azim = azim_
        futures = client.compute(out)
        client.gather(futures)
        t2 = time.time()
        print('Images generation runtime (s): ' + str(np.around(t2-t1, 2)))

    def __write_vtk_frame(self, out_name, out_dir, time_point, scalar_data=None):
        '''
        Format in which a cell morphology (in a singular time point) (color-coded with scalar data, by default membrane voltage) 
        is saved to be visualized in paraview
        '''
        def header_(out_name_=out_name):
            h = "# vtk DataFile Version 4.0\n{}\nASCII\nDATASET POLYDATA\n".format(
                out_name_)
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
            # LINES n_cells total_amount_integers
            of.write("LINES {} {}\n".format(
                len(self.line_pairs), 3*len(self.line_pairs)))
            # e.g. 2 16 765 means index 765 and 16 define a single line, the leading 2 defines the amount of points that define a line
            of.write(line_pairs_str_(self.line_pairs))

            # Diameters
            of.write("POINT_DATA {}\nSCALARS Diameter float 1\nLOOKUP_TABLE default\n".format(
                len(self.diameters)))  # SCALARS name_of_data data_type n_components
            of.write(diameter_str_(self.diameters))

            # Scalar data (as of now only membrane voltages)
            if scalar_data is not None:
                of.write("CELL_DATA {}\nSCALARS Vm float 1\nLOOKUP_TABLE default\n".format(
                    len(self.morphology)))
                of.write(membrane_voltage_str_(scalar_data))

    def __get_interactive_cell(self, background_color="rgb(180,180,180)", renderer="notebook_connected"):
        ''' 
        Setup plotly for rendering in notebooks. Shows an interactive 3D render of the Cell with NO data overlayed.

        Args:
            - background_color: just some grey by default
            - renderer

        Returns:
            plotly.graph_objs._figure.Figure: an interactive figure. Usually added to a ipywidgets.VBox object
        '''
        transparent = "rgba(0, 0, 0, 0)"
        ax_layout = dict(
            backgroundcolor=transparent,
            gridcolor=transparent,
            showbackground=True,
            zerolinecolor=transparent,
            visible=False
        )

        py.init_notebook_mode()
        pio.renderers.default = renderer

        # Create figure
        fig = px.scatter_3d(
            self.morphology, x="x", y="y", z="z",
            hover_data=["section", "diameter"],
            size="diameter")
        fig.update_traces(marker=dict(line=dict(width=0))
                          )  # remove outline of markers
        fig.update_layout(scene=dict(
            xaxis=ax_layout,
            yaxis=ax_layout,
            zaxis=ax_layout,
            bgcolor=transparent  # turn off background for just the scene
        ),
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    def __get_interactive_plot_with_scalar_data(self, scalar_data_keyword, t_start=None, t_end=None, t_step=None, vmin=None, vmax=None, color_map='jet',
                                                background_color="rgb(180,180,180)", renderer="notebook_connected"):
        """This is the main function to set up an interactive plot with scalar data overlayed.

        Args:
            scalar_data_keyword (str, optional): Scalar data to overlay on interactive plot. Defaults to None.
            t_start (float/int, optional): Starting time of the interactive visualisation. Defaults to None.
            t_end (float/int, optional): End time of the interactive visualisation. Defaults to None.
            t_step (float/int, optional): Time step between consecutive time points. Defaults to None.
            vmin (float, optional): Minimum voltage to show. Defaults to None.
            vmax (float, optional): Maximum voltage to show. Defaults to None.
            background_color (str, optional): Background color of plot. Defaults to "rgb(180,180,180)".
            renderer (str, optional): Type of backend renderer to use for rendering the javascript/HTML VBox. Defaults to "notebook_connected".

        Returns:
            ipywidgets.VBox object: an interactive render of the cell.
        """
        if scalar_data_keyword.lower() in ("voltage", "membrane voltage", "vm"):
            self.__calc_voltage_timeseries()
            scalar_data_per_section = np.array([self.__get_voltages_at_timepoint(
                t) for t in np.arange(self.t_start, self.t_end+self.dt, self.dt)]).T
            round_floats = 2
            scalar_data_per_time = {t: np.round(
                self.voltage_timeseries[t_idx], round_floats) for t_idx, t in enumerate(self.times_to_show)}

        else:
            assert scalar_data_keyword.lower() in (e.lower()
                                                   for e in self.possible_scalars), "Keyword {} is not associated with membrane voltage, and does not appear in the possible ion dynamic keywords: {}".format(scalar_data_keyword, self.possible_scalars)
            self.__calc_ion_dynamics_timeseries(
                ion_keyword=scalar_data_keyword)
            scalar_data_per_section = np.array([self.__get_ion_dynamic_at_timepoint(
                t, ion_keyword=scalar_data_keyword) for t in np.arange(self.t_start, self.t_end+self.dt, self.dt)]).T
            scalar_data_per_time = {
                t: self.scalar_data[scalar_data_keyword][t_idx] for t_idx, t in enumerate(self.times_to_show)}

        self.__update_times_to_show(t_start, t_end, t_step)
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        sections = self.morphology["section"]

        # Create figure
        fig_cell = self.__get_interactive_cell(
            background_color=background_color, renderer=renderer)
        fig_cell.update_traces(name="morphology")
        fig_cell.update_coloraxes(colorscale=color_map)
        fig_cell.add_traces(
            go.Scatter3d(x=[0], y=[0], z=[0], name="selection",
                         marker=dict(color='yellow'))
        )
        fig_trace = px.line(
            x=np.arange(self.t_start, self.t_end+self.dt, self.dt), y=self.__get_soma_voltage_between_timepoints(self.t_start, self.t_end, self.dt),
            labels={
                "x": "time (ms)",
                "y": scalar_data_keyword},
            title="{} at soma".format(scalar_data_keyword)
        )
        widget_trace = go.FigureWidget(fig_trace.data, fig_trace.layout)

        # create FigureWidget from figure
        widget_morphology = go.FigureWidget(
            data=fig_cell.data, layout=fig_cell.layout)

        # update color scale
        def _update(time_point):
            """
            Function that gets called whenever the slider gets changed. Requests the membrane voltages at a certain time point
            """
            widget_morphology.update_traces(marker={"color": [
                                            "white" if e is None else e for e in scalar_data_per_time[time_point]]}, selector=dict(name="morphology"))
            widget_morphology.layout.title = "{} at time={} ms".format(
                scalar_data_keyword, time_point)
            widget_trace.update_layout(
                shapes=[dict(
                    type='line',
                    yref='y domain', y0=0, y1=1,
                    xref='x', x0=time_point, x1=time_point
                )
                ])
            return fig_cell

        def update_selection_on_click(trace, points, selector):
            point_ind = points.point_inds[-1]
            for e in ['x', 'y', 'z']:
                widget_morphology.data[1][e] = [
                    self.morphology.iloc[point_ind][e]]
            widget_trace.data[0]['y'] = scalar_data_per_section[point_ind]
            widget_trace.layout.title = "{} of point {}, section {}".format(
                scalar_data_keyword, point_ind, int(sections[point_ind]))
            return widget_morphology

        widget_morphology.data[0].on_click(update_selection_on_click)

        # display the FigureWidget and slider with center justification
        slider = interactive(
            _update,
            time_point=widgets.FloatSlider(min=self.t_start, max=self.t_end, step=self.t_step, value=0, layout=Layout(
                width='800px'), background_color=background_color)
        )
        hb = HBox((widget_morphology, widget_trace))
        vb = VBox((hb, slider))
        vb.layout.align_items = 'center'
        return vb

    def __display_interactive_morphology_only_3d(self, background_color="rgb(180,180,180)", highlight_section=None, renderer="notebook_connected"):
        ''' 
        Setup plotly for rendering in notebooks. Shows an interactive 3D render of the Cell with NO data overlayed.
        If you want to overlay scalar data, such as membrane voltage, please use :@function self.display_interactive_voltage_in_morphology_3d:

        Args:
            - background_color: just some grey by default
            - renderer

        Returns:
            ipywidgets.VBox object: an interactive render of the cell.
        '''
        df = self.morphology.copy()
        fig_cell = self.__get_interactive_cell(
            background_color=background_color, renderer=renderer)
        if highlight_section:
            fig_cell.add_traces(
                px.scatter_3d(df[df['section'] == highlight_section], x="x", y="y", z='z',
                              hover_data=["x", "y", "z", "section", "diameter"], size="diameter")
                .update_traces(marker=dict(line=dict(width=0), color='red'))
                .data
            )

        # create FigureWidget from figure
        f = go.FigureWidget(data=fig_cell.data, layout=fig_cell.layout)
        return f

    # Public methods

    def show_morphology_3d(self, save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D

        Args:
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        fig = plt.figure(figsize=(15, 15), dpi=self.dpi)
        ax = plt.axes(projection='3d', proj_type='ortho')
        sections = np.unique(self.morphology['section'].values)
        for sec in sections:
            points = self.morphology.loc[self.morphology['section'] == sec]
            for i in points.index:
                if i != points.index[-1]:
                    linewidth = (
                        points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                    ax.plot3D([points.loc[i]['x'], points.loc[i+1]['x']],
                              [points.loc[i]['y'], points.loc[i+1]['y']],
                              [points.loc[i]['z'], points.loc[i+1]['z']], color='grey', lw=linewidth)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.roll = self.roll
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

        if save != '':
            plt.savefig(save)  # ,bbox_inches='tight')
        if plot:
            plt.show()

    def show_voltage_in_morphology_3d(self, time_point, vmin=None, vmax=None, voltage_legend=True, save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage for a particular time point.

        Args:
            - time_point: time of the simulation which we would like to visualize
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - voltage_legend: whether the voltage legend should appear in the plot
            - save: path where the plot will be saved. If it's empty it will not be saved
            - plot: whether the plot should be shown.
        '''
        self.__assert_has_simulation_data()
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        voltage = self.__get_voltages_at_timepoint(time_point)

        # Plot morphology with colorcoded voltage
        cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
            vmin=self.vmin, vmax=self.vmax), cmap=plt.get_cmap('jet'))
        fig = plt.figure(figsize=(15, 15), dpi=self.dpi)
        ax = plt.axes(projection='3d', proj_type='ortho')
        sections = np.unique(self.morphology['section'].values)
        for sec in sections:
            points = self.morphology.loc[self.morphology['section'] == sec]
            for i in points.index:
                if i != points.index[-1]:
                    color = ((voltage[i]+voltage[i+1])/2 -
                             self.vmin)/(self.vmax-self.vmin)
                    linewidth = (
                        points.loc[i]['diameter']+points.loc[i+1]['diameter'])/2*1.5+0.2
                    ax.plot3D([points.loc[i]['x'], points.loc[i+1]['x']],
                              [points.loc[i]['y'], points.loc[i+1]['y']],
                              [points.loc[i]['z'], points.loc[i+1]['z']], color=mpl.cm.jet(color), lw=linewidth)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')

        # Add legends
        if voltage_legend:
            ax.legend(frameon=False, bbox_to_anchor=(-0.1715, 0.65, 1., .1), fontsize=12,  # -0.1715, 0.75, 1., .1
                      title='Time = {}ms\n\n'.format(np.around(time_point, 1)), title_fontsize=12)
            # 0.64, 0.2, 0.05, 0.5
            cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5])
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=mpl.cm.jet),
                         ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)  # , fraction=0.015, pad=-0.25)
            plt.axis('off')

        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.roll = self.roll
        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

        if save != '':
            plt.savefig(save)  # ,bbox_inches='tight')
        if plot:
            plt.show()

    def show_voltage_synapses_in_morphology_3d(self, time_point, time_show_syn_activ=None, vmin=None, vmax=None, voltage_legend=True, synapse_legend=True,
                                               save='', plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
        are shown for a particular time point.

        Args:
            - Time_point: time of the simulation which we would like to visualize
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - voltage_legend: whether the voltage legend should appear in the plot
            - synapse_legend: whether the synapse activations legend should appear in the plot
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        self.__assert_has_simulation_data()
        if time_show_syn_activ is not None:
            self.time_show_syn_activ = time_show_syn_activ
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        voltage = self.__get_voltages_at_timepoint(time_point)
        synapses = self.__get_synapses_at_timepoint(time_point)
        self.__plot_cell_voltage_synapses_in_morphology_3d(
            voltage, synapses, time_point, voltage_legend=voltage_legend, synapse_legend=synapse_legend, save=save, plot=plot)

    def write_gif_voltage_synapses_in_morphology_3d(self, images_path, out_path, client=None, t_start=None, t_end=None, t_step=None,
                                                    neuron_rotation=None, time_show_syn_activ=None, vmin=None, vmax=None, frame_duration=40,
                                                    voltage_legend=True, synapse_legend=True):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a gif.
        The parameters :param:t_begin, :param:t_end and :param:t_step will define the :param:self.time attribute

        Args:
            - images_path: dir where the images for the gif will be generated
            - out_path: dir where the gif will be generated + name of the gif
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - client: dask client for parallelization
            - neuron_rotation: rotation degrees of the neuron at each frame (in azimuth)
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - frame_duration: duration of each frame in ms
            - voltage_legend: whether the voltage legend should appear in the plot
            - synapse_legend: whether the synapse activations legend should appear in the plot
        '''
        self.__assert_has_simulation_data()
        if client is None:
            raise ValueError(
                "Please provide a dask client object for the client argument")
        self.__update_times_to_show(t_start, t_end, t_step)
        if neuron_rotation is not None:
            self.neuron_rotation = neuron_rotation
        if time_show_syn_activ is not None:
            self.time_show_syn_activ = time_show_syn_activ
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(
            images_path, client, voltage_legend, synapse_legend)
        write_gif_from_images(images_path, out_path, duration=frame_duration)

    def write_video_voltage_synapses_in_morphology_3d(self, images_path, out_path, client=None, t_start=None, t_end=None, t_step=None,
                                                      neuron_rotation=None, time_show_syn_activ=None, vmin=None, vmax=None,
                                                      framerate=12, quality=5, codec='mpeg4', voltage_legend=True, synapse_legend=True):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a video.
        The parameters :param:t_begin, :param:t_end and :param:t_step will define the :param:time attribute

        Args:
            - images_path: dir where the images for the video will be generated
            - out_path: dir where the video will be generated + name of the video
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - client: dask client for parallelization
            - neuron_rotation: rotation degrees of the neuron at each frame (in azimuth)
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - framerate: frames per second
            - quality
            - codec
            - voltage_legend: whether the voltage legend should appear in the plot
            - synapse_legend: whether the synapse activations legend should appear in the plot
        '''
        self.__assert_has_simulation_data()
        if client is None:
            raise ValueError(
                "Please provide a dask client object for the client argument")
        self.__update_times_to_show(t_start, t_end, t_step)
        if neuron_rotation is not None:
            self.neuron_rotation = neuron_rotation
        if time_show_syn_activ is not None:
            self.time_show_syn_activ = time_show_syn_activ
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(
            images_path, client, voltage_legend, synapse_legend)
        write_video_from_images(images_path, out_path,
                                fps=framerate, quality=quality, codec=codec)

    def display_animation_voltage_synapses_in_morphology_3d(self, images_path, client=None, t_start=None, t_end=None, t_step=None,
                                                            neuron_rotation=None, time_show_syn_activ=None, vmin=None, vmax=None,
                                                            voltage_legend=True, synapse_legend=True):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a python animation.
        The parameters :param:t_begin, :param:t_end and :param:t_step will define the :param:self.time attribute


        Args:
            - images_path: path where the images for the gif will be generated
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - client: dask client for parallelization
            - neuron_rotation: rotation degrees of the neuron at each frame (in azimuth)
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - voltage_legend: whether the voltage legend should appear in the plot
            - synapse_legend: whether the synapse activations legend should appear in the plot
        '''
        self.__assert_has_simulation_data()
        if client is None:
            raise ValueError(
                "Please provide a dask client object for the client argument")
        self.__update_times_to_show(t_start, t_end, t_step)
        if neuron_rotation is not None:
            self.neuron_rotation = neuron_rotation
        if time_show_syn_activ is not None:
            self.time_show_syn_activ = time_show_syn_activ
        if vmin is not None:
            self.vmin = vmin
        if vmax is not None:
            self.vmax = vmax
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(
            images_path, client, voltage_legend, synapse_legend)
        display_animation_from_images(images_path, 1, embedded=True)

    def display_interactive_morphology_3d(self, data=None, background_color="rgb(180,180,180)", highlight_section=None, renderer="notebook_connected",
                                          t_start=None, t_end=None, t_step=None, vmin=None, vmax=None, color_map="jet", show=True):
        """This method shows a plot with an interactive cell, overlayed with scalar data (if provided with the data argument).
        The parameters :param:t_begin, :param:t_end and :param:t_step will define the :param:self.time attribute

        Args:
            data (str, optional): Scalar data to overlay on interactive plot. Defaults to None.
            background_color (str, optional): Background color of plot. Defaults to "rgb(180,180,180)".
            highlight_section (int, optional): If no scalar data is provided, this argument can be set to an integer, defining a section to highlight on the interactive plot. Defaults to None.
            renderer (str, optional): Type of backend renderer to use for rendering the javascript/HTML VBox. Defaults to "notebook_connected".
            t_start (float/int, optional): Starting time of the interactive visualisation. Defaults to None.
            t_end (float/int, optional): End time of the interactive visualisation. Defaults to None.
            t_step (float/int, optional): Time step between consecutive time points. Defaults to None.
            vmin (float, optional): Minimum voltage to show. Defaults to None.
            vmax (float, optional): Maximum voltage to show. Defaults to None.

        Returns:
            ipywidgets.VBox object: an interactive render of the cell.
        """
        if data is None:
            f = self.__display_interactive_morphology_only_3d(background_color=background_color, highlight_section=highlight_section, renderer=renderer)
        else:
            f = self.__get_interactive_plot_with_scalar_data(data, t_start=t_start, t_end=t_end, t_step=t_step, vmin=vmin, vmax=vmax,
                                                             color_map=color_map, background_color=background_color, renderer=renderer)
        
        return f.show() if show else f

    def display_interactive_voltage_in_morphology_3d(self, t_start=None, t_end=None, t_step=None, vmin=None, vmax=None, color_map='jet', background_color="rgb(180,180,180)", renderer="notebook_connected"):
        ''' 

        Setup plotly for rendering in notebooks. Shows an interactive 3D render of the Cell with the following data overlayed:

        - Membrane voltage
        - Section ID
        - 3D Coordinates

        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - color_map: voltage color map
            - background_color: just some grey by default
            - renderer

        Returns:
            ipywidgets.VBox object: an interactive render of the cell.

        Todo:
            add synapse activations!
            add dendritic and somatic AP as secondary subplot
        '''
        return self.display_interactive_morphology_3d(data="voltage", t_start=t_start, t_end=t_end, t_step=t_step, vmin=vmin, vmax=vmax, color_map=color_map, background_color=background_color, renderer=renderer)

    def display_interactive_ion_dynamics_in_morphology_3d(self, ion_keyword="Ca_HVA.ica", t_start=None, t_end=None, t_step=None, vmin=None, vmax=None, color_map='jet',
                                                          background_color="rgb(180,180,180)", renderer="notebook_connected"):
        ''' 
        Setup plotly for rendering in notebooks. Shows an interactive 3D render of the Cell with the following data overlayed:

        - Ion channel dynamics
        - Section ID
        - 3D Coordinates

        Args:
            - ion_dynamics: keyword to specify which ion channel dynamic to plot. Either the name of a rangeVar
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - vmin: min voltages colorcoded in the cell morphology
            - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
            - color_map: voltage color map
            - background_color: just some grey by default
            - renderer
            
        Returns:
            ipywidgets.VBox object: an interactive render of the cell.
        '''
        return self.display_interactive_morphology_3d(data=ion_keyword, t_start=t_start, t_end=t_end, t_step=t_step, vmin=vmin, vmax=vmax, color_map=color_map,
                                                      background_color=background_color, renderer=renderer)

    def write_vtk_frames(self, out_name="frame", out_dir=".", t_start=None, t_end=None, t_step=None):
        '''
        Format in which a cell morphology timeseries (color-coded with voltage) is saved to be visualized in paraview

        Todo:
            remove duplicate rows used to connect sections

        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - out_name: name of the file (not path, the file will be generated in out_dir)
            - out_dir: path where the images for the gif will be generated
        '''
        self.__update_times_to_show(t_start, t_end, t_step)
        self.__calc_voltage_timeseries()

        progress = self.times_to_show
        if HAS_TQDM:
            progress = tqdm(t, desc="Writing vtk frames to {}".format(out_dir))
        for t in progress:
            self.__write_vtk_frame(
                out_name, out_dir, scalar_data=self.voltage_timeseries[t])


@dask.delayed
def plot_cell_voltage_synapses_in_morphology_3d(morphology, voltage, synapses, time_point, save, population_to_color_dict,
                                                azim=0, dist=10, elev=30, roll=0, vmin=-75, vmax=-30, voltage_legend=True,
                                                synapse_legend=True, time_offset=0, dpi=72):
    '''
    Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
    are shown for a particular time point.
    Dask delayed function useful for parallelization of images generation. This dask delayed function cannot be part of the
    visualization class, dask does not allow it because this class has a cell object as an attribute and dask cannot serialize it,
    if the cell object wasn't an attribute this function could be a class method.
    
    Todo:
        find for a possible solution.

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
        - vmax: max voltages colorcoded in the cell morphology (the lower vmax is, the stronger the APs are observed)
        - voltage_legend: whether the voltage legend should appear in the plot
        - synapse_legend: whether the synapse activations legend should appear in the plot
    '''
    # Plot morphology with colorcoded voltage
    cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
        vmin=vmin, vmax=vmax), cmap=plt.get_cmap('jet'))
    fig = plt.figure(figsize=(15, 15), dpi=dpi)
    ax = plt.axes(projection='3d', proj_type='ortho')
    sections = np.unique(morphology['section'].values)
    for sec in sections:
        points = morphology.loc[morphology['section'] == sec]
        for i in points.index:
            if i != points.index[-1]:
                color = ((voltage[i]+voltage[i+1])/2 - vmin)/(vmax-vmin)
                linewidth = (points.loc[i]['diameter'] +
                             points.loc[i+1]['diameter'])/2*1.5+0.2
                ax.plot3D([points.loc[i]['x'], points.loc[i+1]['x']],
                          [points.loc[i]['y'], points.loc[i+1]['y']],
                          [points.loc[i]['z'], points.loc[i+1]['z']], color=mpl.cm.jet(color), lw=linewidth)

    # Plot synapse activations
    for population in synapses.keys():
        for synapse in synapses[population]:
            color = population_to_color_dict[population]
            ax.scatter3D(synapse[0], synapse[1], synapse[2],
                         color=color, edgecolors='grey', s=75)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')

    # Add legends
    if synapse_legend:
        for key in population_to_color_dict.keys():
            if key != 'inactive':
                ax.scatter3D([], [], [], color=population_to_color_dict[key],
                             label=key, edgecolor='grey', s=75)
    if voltage_legend:
        cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5])  # 0.64, 0.2, 0.05, 0.5
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=mpl.cm.jet),
                     ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)  # , fraction=0.015, pad=-0.25)
        plt.axis('off')
    # Time always visible
    ax.legend(frameon=False, bbox_to_anchor=(-0.1715, 0.65, 1., .1), fontsize=12,  # -0.1715, 0.75, 1., .1
              title='Time = {}ms\n\n'.format(np.around(time_point-time_offset, 1)), title_fontsize=12)

    ax.azim = azim
    ax.dist = dist
    ax.elev = elev
    ax.roll = roll
    ax.set_box_aspect(
        [ub - lb for lb, ub in (getattr(ax, 'get_{}lim'.format(a))() for a in 'xyz')])

    plt.savefig(save)  # ,bbox_inches='tight')
    plt.close()
