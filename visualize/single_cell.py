import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import dask
import time
import jinja2
import IPython
import glob
import warnings
from barrel_cortex import inhibitory
warnings.filterwarnings("default", category=ImportWarning, module=__name__)  # let ImportWarnings show up when importing this module through Interface
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    warnings.warn("tqdm not found in the current environment. No progressbars will be displayed", ImportWarning)
    HAS_TQDM = False
from ipywidgets import interactive, VBox, widgets, Layout
try:
    import plotly.offline as py
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    warnings.warn("Plotly could not be imported. Interactive visualisations will not work.", ImportWarning)
import pandas as pd


class CellVisualizer:
    """
    @TODO: make main dataformat a pandas dataframe

    This class initializes from a cell object and extracts relevant Cell data to a format that lends itself easier to plotting.
    It contains useful methods for either plotting a cell morphology, the voltage along its body and its synaptic inputs.
    This can be visualized in static images or as time series (videos, gif, animation, interactive window).
    Also, the relevant cell information can also be exported to .vtk format for further visualization or interaction.
    """
    def __init__(self, cell):
        """
        given a Cell object, this class initializes an object that is easier to work with
        """
        # Colors in which the synaptic input are going to be shown
        self.population_to_color_dict = {'INT': 'black', 'L4ss': 'orange', 'L5st':'cyan', 'L5tt':'lime',
                     'L6CC':'yellow', 'VPM':'red', 'L23':'magenta', 'inactive':'white'}
        
        # Angles in which our 3D plots are shown
        self.azim = 0  # Azimuth of the projection in degrees.
        self.dist = 10 # Distance of the camera to the object.
        self.elev = 30 # Elevation of the camera above the equatorial plane in degrees.
        
        # Max and min voltages colorcoded in the cell morphology
        self.vmin = -75 # mV
        self.vmax = -40 # mV
        
        # Time in the simulation during which a synapse activation is shown during the visualization
        self.time_show_syn_activ = 2 # ms
        
        # Cell object
        self.cell = cell
        
        # The different sections that form the cell morphology. Sections are defined as the regions of cell morphology
        # between 2 branching points / end of a branch.
        # A single section is formed by a list of 3d points that define the cell structure in the 3d space.
        # Each point has an x, y and z coordinate and a diameter (a point is defined as a 4-element tuple).
        self.sections = []
        
        # If simulations have been run, this is list of time points which we'd like to visualize.
        self.time = np.empty(0)
        
        # List contaning the voltage of the cell during a timeseries. Each element corresponds to a time point.
        # Each element of the list contains n elements, being n the number of sections of the cell morphology.
        # These elements are also lists of m elements, being m the number of points in each section, and the value of
        # those lists are the voltage at each point of the cell morphology.
        self.voltage_timeseries = []
        
        # List containing the synapse activations during a timeseries (Similarly to voltage_timeseries). 
        # Each element corresponds to a time point. Each element is a dictionary where each key is the type of
        # input population and the value is the list of active synapses for that type of population. The list contains
        # the 3d coordinates where each active synapse is located.
        self.synapses_timeseries = []
        
        # Gather the necessary information to plot the cell morphology. This info is always necessary to plot the cell.
        self._gather_morphology()
        
        ## @TODO: merge these attributes with the previous ones. There is repeated data about the cell morphology and voltage.
        self.line_pairs = []
        self.points = np.empty(0)
        self.diameters = []  # defined per point
        self.n_time_points = len(self.time)
        self.membrane_voltage = {t: [] for t in range(self.n_time_points)}  # each key is a time point INDEX. Each value is a list of membrane voltages of the entire neuron for that time point
        self.section_indices = []
        self.section_labels = []
        
    def __parse_cell(self):
        """
        @TODO: create checks so that the cells is not parsed again if it has already been parsed
        Given a cell object, this method parses the data to export it easier to vtk of df formats
        """
        def __update_segment_indices_(point_index_, section_, segment_limits_, next_seg_flag_, current_seg_):
            """
            Updates various segment info depending on the current segment, segment limits and section.
            Arguments:
                - point_index: int: index of a point in some segment in some section
                - section_: Section: Section attribute of a Cell instance.
                - segment_limits_: ???
                - next_seg_flag_: bool: flag to check if iteration should go to next section
                current_seg_: int: Index of current segment. Augments one at a time, resets when jumping to next section.
            """
            if point_index_ != 0: # Plot lines that join points within the same section
                if next_seg_flag_:
                    current_seg_ += 1
                    next_seg_flag_ = False 
                if section_.relPts[point_index_-1] < segment_limits_[current_seg_+1] < section_.relPts[point_index_]:
                    if section_.relPts[point_index_] - segment_limits_[current_seg_+1] > segment_limits_[current_seg_+1] - section_.relPts[point_index_-1]:
                        current_seg_ += 1
                    else:
                        next_seg_flag_ = True
            else:
                pass
            return current_seg_, next_seg_flag_

        def __construct_segment_limits(section):
            """
            Given a certain cell section, this method construct the limits
            """
            segs_limits = [0]
            for j, seg in enumerate(section):
                # seg.x is the 1d coordinate between each section. X corresponds to some point where the voltage was measured
                segs_limits.append(segs_limits[j]+(seg.x-segs_limits[j])*2)
            return segs_limits
        
        # iterate sections and extract data per section. Add this data to flat lists
        section_n_points = []
        for lv, sec in enumerate(self.cell.sections):  

            if sec.label in ['Soma', 'AIS', 'Myelin']:  # skip these sections
                continue

            segs_limits = __construct_segment_limits(sec)
            current_seg = 0
            next_seg_flag = False 
            n_prev_points = int(np.sum(section_n_points))

            # append first point of the section
            self.points.append(sec.pts[0])
            self.diameters.append(sec.diamList[0])
            self.section_indices.append(lv)
            self.section_labels.append(sec.label)
            for t in range(self.n_time_points):
                    self.membrane_voltage[t].append(sec.recVList[current_seg][t])
            # append the rest of the section
            for i, pt in enumerate(sec.pts[1:]):  # append consecutive points plus connection
                self.points.append(pt)
                self.diameters.append(sec.diamList[i])  # save diameter of this section to instance
                self.section_indices.append(lv)
                self.section_labels.append(sec.label)
                index = n_prev_points + i
                self.line_pairs.append([index, index+1])
                current_seg, next_seg_flag = __update_segment_indices_(i, sec, segs_limits, next_seg_flag, current_seg)
                for t in range(self.n_time_points):
                    self.membrane_voltage[t].append(sec.recVList[current_seg][t])
            section_n_points.append(len(sec.pts))
        self.points = np.array(self.points)

        """ connect sections
        TODO: this causes paraview to segfault. Perhaps lines can't share a point?
        for lv, sec in enumerate(cell.sections):
            if sec.parent is not None:
                parent_section_index = cell.sections.index(sec.parent)  # index of parent section in section list
                pt_index_1 = I.np.sum(section_n_points[:parent_section_index + 1]) - 1  # index of last point of the parent section in points list
                pt_index_2 = I.np.sum(section_n_points[:lv]) # index of first point of the current section in points list
                diameters.append(sec.parent.diamList[-1])  # diameter of section
                if pt_index_2 > len(points):
                    raise ValueError("Index cannot be larger than amount of points")
                line_pair = [pt_index_1, pt_index_2]
                if line_pair not in line_pairs and line_pair[::-1] not in line_pairs:
                    line_pairs.append([pt_index_1, pt_index_2])  # add connection piece between sections
        """
        
    def __to_df(self, t=0, round_floats=2):
        """
        Constructs a dataframe with the following columns and data: ["x", "y", "z", "membrane voltage", "section index", "diameter"].
        The membrane voltage is the voltage at time t=0 be default.
        Args:
            - self: Cell
            - t: int: time at which the membrane voltage should be defined.
            - round: int: round the membrane voltages to this amount of numbers after the comma
        Returns:
            df: pandas.DataFrame
        """
        x, y, z = self.points.T
        df = pd.DataFrame.from_dict({
            "x": np.round(x, round_floats), "y": np.round(y, round_floats), "z": np.round(z, round_floats), 
            "membrane voltage": np.round(self.membrane_voltage[t], round_floats),
            "section index": self.section_indices,
            "diameter": np.round(self.diameters, round_floats),
            "section label": self.section_labels})
        return df
        
    def __write_vtk_frame(self, out_name, out_dir, time_point):
        '''
        Format in which a cell morphology (in a simple time point) (color-coded with voltage) 
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
        
        def scalar_data_str_(scalar_name_):
            """
            given the name of
            """
            v_string = ""
            for v in self.scalar_data[scalar_name_]:
                v_string += str(v)
                v_string += "\n"
            return v_string
        
        # write out all data to .vtk file
        with open(out_dir+"/"+out_name+"_{:06d}.vtk".format(time_point), "w+", encoding="utf-8") as of:
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
            for name, data in self.scalar_data.items():
                of.write("CELL_DATA {}\nSCALARS Vm float 1\nLOOKUP_TABLE default\n".format(len(data)))
                of.write(scalar_data_str_(name))
                
    def write_vtk_frames(self, out_name, out_dir):
        '''
        Format in which a cell morphology timeseries (color-coded with voltage) is saved to be visualized in paraview
        '''
        self.__parse_cell()
        progress = range(self.n_time_points)
        if HAS_TQDM:
            progress = tqdm(t, desc="Writing vtk frames to {}".format(out_dir))
        for t in progress:
                self.__write_vtk_frame(out_name, out_dir, time_point=t)
        
    
    def __display_animation(files, interval=10, style=False, animID = None, embedded = False):
        '''
        @TODO: resolve module not found errors
        creates an IPython animation out of files specified in a globstring or a list of paths.

        animID: unique integer to identify the animation in the javascript environment of IPython
        files: globstring or list of paths
        interval: time interval between frames

        CAVEAT: the paths need to be relative to the location of the ipynb / html file, since
        the are resolved in the browser and not by python'''
        if animID is None:
            animID = np.random.randint(10000000000000) # needs to be unique within one ipynb
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
        template = env.get_template(html_template)

        if isinstance(files, str):
            if os.path.isdir(files): # folder provieded --> convert to globstring
                files = os.path.join(files, '*.png')
            listFrames = sorted(glob.glob(files))
        else:
            listFrames = files
        if embedded:
            listFrames = [_load_base64(f) for f in listFrames]
        htmlSrc = template.render(ID=animID, listFrames=listFrames, interval=interval, style=style)
        IPython.display.display(IPython.display.HTML(htmlSrc))
        
    def __create_gif(self, name, files, duration=40):
        '''
        Creates a gif from a set of images
        Args:
            - Name: path where the gif will be located + name of the gif
            - Files: list of the images that will be used for the gif generation
            - Duration: duration of each frame in ms
        '''
        frames = [] # Create the frames
        for file in files:
            new_frame = Image.open(file)
            frames.append(new_frame)

        # Save into a GIF file that loops forever   
        frames[0].save(name, format='GIF',
                       append_images=frames[1:],
                       save_all=True,
                       duration=duration, loop=0)
        
    def __match_model_celltype_to_PSTH_celltype(self, celltype):
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
        
    def show_cell_morphology_3d(self, save='',plot=True):
        '''
        Creates a python plot of the cell morphology in 3D
        Args:
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        fig = plt.figure(figsize = (15,15))
        ax = plt.axes(projection='3d')
        for lv, sec in enumerate(self.sections):  
            for i,pt in enumerate(sec): 
                if i!= 0:
                    ax.plot3D([sec[i-1][0],sec[i][0]], [sec[i-1][1],sec[i][1]], [sec[i-1][2],sec[i][2]], 
                               color='grey' ,lw=sec[i][3]*1.5+0.2) 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

        if save != '':
            plt.savefig(save)#,bbox_inches='tight')
        if plot:
            plt.show()
        
    def show_cell_voltage_in_morphology_3d(self, time_point,legends=True,save='',plot=True):
        '''
        Creates a python plot of the cell morphology in 3D color-coded with voltage for a particular time point.
        Args:
            - Time_point: time of the simulation which we would like to visualize
            - Legends: whether the voltage legend should appear in the plot
            - Save: path where the plot will be saved. If it's empty it will not be saved
            - Plot: whether the plot should be shown.
        '''
        voltage = self._gather_voltage_at_timepoint(time_point)
        # Plot morphology with colorcoded voltage
        cmap=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=plt.get_cmap('jet'))
        fig = plt.figure(figsize = (15,15))
        ax = plt.axes(projection='3d')
        for lv, sec in enumerate(self.sections):  
            for i,pt in enumerate(sec): 
                if i!= 0:
                    color = (voltage[lv][i-1] - self.vmin)/(self.vmax-self.vmin)#vmin = -75; vmax = -40
                    ax.plot3D([sec[i-1][0],sec[i][0]], [sec[i-1][1],sec[i][1]], [sec[i-1][2],sec[i][2]], 
                               color=mpl.cm.jet(color) ,lw=sec[i][3]*1.5+0.2) 
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
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

        if save != '':
            plt.savefig(save)#,bbox_inches='tight')
        if plot:
            plt.show()
        
    def show_cell_voltage_synapses_in_morphology_3d(self, time_point, time_show_syn_activ=2, legends=True,save='',plot=True):
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
        voltage = self._gather_voltage_at_timepoint(time_point)
        synapses = self._gather_synapses_at_timepoint(time_point)
        self.__plot_cell_voltage_synapses_in_morphology_3d(voltage,synapses,time_point,legends=legends,save=save,plot=plot)
        
    def _gather_morphology(self):
        '''
        Retrieve cell MORPHOLOGY from cell object.
        Fills the self.sections attribute
        '''
        self.sections = []
        for lv, sec in enumerate(self.cell.sections):
            points = []
            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue
            # Point that belongs to the previous section
            x = sec.parent.pts[-1][0]
            y = sec.parent.pts[-1][1]
            z = sec.parent.pts[-1][2]
            d = sec.parent.diamList[-1]
            points.append((x, y, z, d))

            for i,pt in enumerate(sec.pts):
                # Points within the same section
                x = pt[0]
                y = pt[1]
                z = pt[2]
                d = sec.diamList[i]
                points.append([x, y, z, d])
            self.sections.append(points)
        
    def _gather_voltage_at_timepoint(self, time_point):
        '''
        Retrieves the VOLTAGE along the whole cell morphology from cell object at a particular time point.
        Args:
         - time_point: time point from which we want to gather the voltage
        '''
        n_sim_point = int(np.where(np.isclose(self.cell.t, time_point))[0][0]) 
        # or n_sim_point = int(time_point/dt_frames), dt_frames = self.cell.t[1]-self.cell.t[0]
        voltage_sections = []
        for lv, sec in enumerate(self.cell.sections):
            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue

            # Compute segments limits (voltage is measure at the middle of each segment, not section)
            segs_limits = [0]
            for j,seg in enumerate(sec):
                segs_limits.append(segs_limits[j]+(seg.x-segs_limits[j])*2)

            # Map the segments to section points to assign a voltage to each line connecting 2 points
            current_seg = 0
            next_seg_flag = False 
            voltage_points = []
            for i,pt in enumerate(sec.pts):
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
            voltage_sections.append(voltage_points)
        return voltage_sections
        
    def _gather_timeseries_voltage(self):
        '''
        Retrieves VOLTAGE along the whole cell body during a set of time points (specified in self.time).
        Fills the self.voltage_timeseries attribute.
        '''
        if len(self.voltage_timeseries)!=0:
            return # We have already retrieved the voltage timeseries
        
        t1 = time.time()
        for time_point in self.time: # For each frame of the video/animation
            voltage_sections = self._gather_voltage_at_timepoint(time_point)
            self.voltage_timeseries.append(voltage_sections)
        t2 = time.time()
        print('Voltage retrieval runtime (s): ' + str(np.around(t2-t1,2)))
        
    def _gather_synapses_at_timepoint(self, time_point):
        '''
        Retrieves the SYNAPSE ACTIVATIONS at a particular time point.
        Args:
         - time_point: time point from which we want to gather the synapse activations
        '''
        synapses = {'INT': [], 'L4ss': [], 'L5st':[], 'L5tt':[], 'L6CC':[], 'VPM':[], 'L23':[]}
        for population in self.cell.synapses.keys():
            for synapse in self.cell.synapses[population]:
                for spikeTime in synapse.preCell.spikeTimes:
                    if time_point-self.time_show_syn_activ < spikeTime < time_point+self.time_show_syn_activ:
                        population_name = self.__match_model_celltype_to_PSTH_celltype(population)
                        synapses[population_name].append([synapse.coordinates[0],
                                                          synapse.coordinates[1],
                                                          synapse.coordinates[2]])
        return synapses
        
    def _gather_timeseries_synapses(self): 
        '''
        Retrieves the SYNAPSE ACTIVATIONS during a set of time points (specified in self.time).
        Fills the self.synapses_timeseries attribute.
        '''
        if len(self.synapses_timeseries)!=0:
            return # We have already retrieved the synapses timeseries
        
        t1 = time.time()
        for time_point in self.time: # For each frame of the video/animation
            synapses = self._gather_synapses_at_timepoint(time_point)
            self.synapses_timeseries.append(synapses)
        t2 = time.time()
        print('Synapses retrieval runtime (s): ' + str(np.around(t2-t1,2)))
        
    def __plot_cell_voltage_synapses_in_morphology_3d(self, voltage, synapses, time_point, legends=True, save='', plot=True):
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
        fig = plt.figure(figsize = (15,15))
        ax = plt.axes(projection='3d')
        for lv, sec in enumerate(self.sections):  
            for i,pt in enumerate(sec): 
                if i!= 0:
                    color = (voltage[lv][i-1] - self.vmin)/(self.vmax-self.vmin)
                    ax.plot3D([sec[i-1][0],sec[i][0]], [sec[i-1][1],sec[i][1]], [sec[i-1][2],sec[i][2]], 
                               color=mpl.cm.jet(color) ,lw=sec[i][3]*1.5+0.2) 

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
                      title='Time = {}ms\n\n'.format(np.around(time_point,1)), title_fontsize=12)
            cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5]) # 0.64, 0.2, 0.05, 0.5
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=self.vmin, vmax=self.vmax), cmap=mpl.cm.jet),
                 ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)#, fraction=0.015, pad=-0.25)
            plt.axis('off')

        ax.azim = self.azim
        ax.dist = self.dist
        ax.elev = self.elev
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])

        if save != '':
            plt.savefig(save)#,bbox_inches='tight')
        if plot:
            plt.show()
        else:
            plt.close()
        
    def __timeseries_images_cell_voltage_synapses_in_morphology_3d(self, t_start,t_end,t_step,path, client):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. These images will then be used for a time-series visualization (video/gif/animation)
        and in each image the neuron rotates a bit (3 degrees) over its axis
        Args:
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            These parameters will define the self.time attribute
        '''
        if os.path.exists(path):
            if os.listdir(path):
                print('Images already generated, they will not be generated again. Please, change the path name or delete the current one.')
                return
        else:
            os.mkdir(path)
        
        # Gathers the voltage and synapse activations time series. 
        # Then images are generated for each specified time step.
        new_time = np.arange(t_start,t_end,t_step)
        if len(self.time) != 0:
            print(self.time)
            print(new_time)
            if len(self.time)==len(new_time):
                if (self.time != new_time).any():
                    self.voltage_timeseries = []
                    self.synapses_timeseries = []
            else:
                self.voltage_timeseries = []
                self.synapses_timeseries = []
        self.time = new_time
        self._gather_timeseries_voltage()
        self._gather_timeseries_synapses()
        
        out = []
        azim_ = self.azim
        count = 0
        
        # Create images for animation/video
        t1 = time.time()
        for voltage,synapses in zip(self.voltage_timeseries, self.synapses_timeseries):
            time_point = self.time[count]
            count += 1
            filename = path+'/{0:0=5d}.png'.format(count)
            out.append(plot_cell_voltage_synapses_in_morphology_3d(
                        sections=self.sections, voltage=voltage, synapses=synapses, 
                        time_points=time_point, save=filename, population_to_color_dict=self.population_to_color_dict,
                        azim=self.azim, dist=self.dist, elev=self.elev, vmin=self.vmin, vmax=self.vmax,
                        legends=True, plot=False))
            self.azim += 3
        self.azim = azim_
        futures = client.compute(out)
        client.gather(futures)
        t2 = time.time()
        print('Images generation runtime (s): ' + str(np.around(t2-t1,2)))
        
    def gif_cell_voltage_synapses_in_morphology_3d(self, images_path, name,t_start,t_end,t_step,
                                                   time_show_syn_activ=2, frame_duration=40, client=None):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a gif.
        Args:
            - images_path: path where the images for the gif will be generated
            - name: name of the gif (not path, the gif will be generated in the same path as the images)
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            - frame_duration: duration of each frame in ms
            t_start, t_end and t_step will define the self.time attribute
        '''
        if client is None:
            raise 
        self.time_show_syn_activ = time_show_syn_activ
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(t_start,t_end,t_step,images_path, client)
        files = [os.path.join(images_path, f) for f in os.listdir(images_path)]
        files.sort()
        self.__create_gif(os.path.join(images_path,name),files,frame_duration)
        
    def video_cell_voltage_synapses_in_morphology_3d(self, images_path, name,t_start,t_end,t_step,time_show_syn_activ=2, client=None):
        '''
        @TODO: to be implemented
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a video.
        Args:
            - images_path: path where the images for the gif will be generated
            - name: name of the gif (not path, the gif will be generated in the same path as the images)
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            t_start, t_end and t_step will define the self.time attribute
        '''
        self.time_show_syn_activ = time_show_syn_activ
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(t_start,t_end,t_step,images_path)
        print('Function not implemented yet!!')
        
    def animation_cell_voltage_synapses_in_morphology_3d(self, images_path,t_start,t_end,t_step,time_show_syn_activ=2):
        '''
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into a python animation.
        Args:
            - images_path: path where the images for the gif will be generated
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            t_start, t_end and t_step will define the self.time attribute
        '''
        self.time_show_syn_activ = time_show_syn_activ
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(t_start,t_end,t_step,images_path)
        self.__display_animation(images_path, 1, embedded=True)
        
    def interactive_cell_voltage_synapses_in_morphology_3d(self, images_path,t_start,t_end,t_step,time_show_syn_activ=2):
        ''' 
        @TODO: merge this method with self.plot_interactive_3d
        @TODO: to be implemented
        Creates a set of images where a neuron morphology color-coded with voltage together with synapse activations are
        shown for a set of time points. In each image the neuron rotates a bit (3 degrees) over its axis.
        These images are then put together into an interactive python window.
        Args:
            - images_path: path where the images for the gif will be generated
            - t_start: start time point of our time series visualization
            - t_end: last time point of our time series visualization
            - t_step: time between the different time points of our visualization
            - time_show_syn_activ: Time in the simulation during which a synapse activation is shown during the visualization
            t_start, t_end and t_step will define the self.time attribute
        '''
        self.time_show_syn_activ = time_show_syn_activ
        self.__timeseries_images_cell_voltage_synapses_in_morphology_3d(t_start,t_end,t_step,images_path)
        print('Function not implemented yet!!')
        
    def plot_interactive_3d(self, downsample_time=10, round_floats=2, renderer="notebook_connected"):
        """
        @TODO: merge this method with self.interactive_cell_voltage_synapses_in_morphology_3d
        Setup plotly for rendering in notebooks. Shows an interactive 3D render of the Cell with the following data overlayed:
        - Membrane voltage
        - Section name and section ID (not segment ID though)
        - Coordinates
        Args:
            - downsample_time: int or float: Must be larger or equal to 1. Downsample the timesteps with this factor such that only n_timesteps//downsample_time are actually plotted out
            - round: int: round the membrane voltage values
        Returns:
            ipywidgets.VBox object: an interactive render of the cell.
        """
        self.__parse_cell()
        py.init_notebook_mode()
        pio.renderers.default = renderer
        # Initialize a dataframe. This may seem inefficient, but plotly does this anyways whenever you pass data. 
        # Might as well explicitly do it yourself with more control
        df = self.__to_df(t=0, round_floats=round_floats)
        
        # Create figure
        fig = px.scatter_3d(
            df, x="x", y="y", z="z", 
            hover_data=["section index", "diameter"], hover_name="section label", 
            color="membrane voltage", range_color=[-80, 30],
            size="diameter")
        fig.update_traces(marker = dict(line = dict(width = 0)))  # remove outline of markers
        fig.update_layout(coloraxis_colorbar=dict(title="Membrane voltage (mV)"))

        # create FigureWidget from figure
        f = go.FigureWidget(data=fig.data, layout=fig.layout)

        # update color scale
        def _update(frame):
            """
            Function that gets called whenever the slider gets changed. Requests the membrane voltages at time point :param frame:, corresponding to time point
            t = frame * delta t
            """
            dt = self.time[1] - self.time[0]
            f.update_traces(marker={"color": np.round(self.membrane_voltage[frame*downsample_time], round_floats)})
            f.layout.title = "Membrane voltage at time={} ms".format(round(frame*dt, 4))
            return fig

        # display the FigureWidget and slider with center justification
        slider = interactive(
            _update, 
            frame = widgets.IntSlider(min=0, max=self.n_time_points-1, step=downsample_time, value=0, layout=Layout(width='800px'))
            )
        vb = VBox((f, slider))
        vb.layout.align_items = 'center'
        return vb

@dask.delayed
def plot_cell_voltage_synapses_in_morphology_3d(sections,voltage,synapses, time_point, save, population_to_color_dict,
                                                 azim=0, dist=10, elev=30, vmin=-75, vmax = -30, legends=True):
    '''
    Creates a python plot of the cell morphology in 3D color-coded with voltage, and where the synapse activations
    are shown for a particular time point.
    Dask delayed function useful for parallelization of images generation. This dask delayed function cannot be part of the
    visualization class, dask does not allow it. @TODO: find for a possible solution.
    Args:
        - sections: list of sections of cell morphology. Each section should contain a set of points (x, y, z, diameter).
        - voltage: voltage of the neuron along its morphology for a particular time point. It contains a list of sections,
          where each section contains a list of voltage points in each point of the neuron morphology.
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
    fig = plt.figure(figsize = (15,15))
    ax = plt.axes(projection='3d')
    for lv, sec in enumerate(sections):  
        for i,pt in enumerate(sec): 
            if i!= 0:
                color = (voltage[lv][i-1] - vmin)/(vmax-vmin)
                ax.plot3D([sec[i-1][0],sec[i][0]], [sec[i-1][1],sec[i][1]], [sec[i-1][2],sec[i][2]], 
                           color=mpl.cm.jet(color) ,lw=sec[i][3]*1.5+0.2) 

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
                  title='Time = {}ms\n\n'.format(np.around(time_point,1)), title_fontsize=12)
        cbaxes = fig.add_axes([0.64, 0.13, 0.05, 0.5]) # 0.64, 0.2, 0.05, 0.5
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=mpl.cm.jet),
             ax=cbaxes, orientation='vertical', label='mV', fraction=0.2)#, fraction=0.015, pad=-0.25)
        plt.axis('off')
        
    ax.azim = azim
    ax.dist = dist
    ax.elev = elev
    ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
    
    plt.savefig(save)#,bbox_inches='tight')
    plt.close()
    