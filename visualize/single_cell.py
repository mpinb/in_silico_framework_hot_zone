import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CellVisualizer:
    """
    This class initializes from a cell object and transform the Cell data to something that lends itself easier to plotting.
    It contains useful methods for either plotting a cell, or writing it out to .vtk frames.
    """
    def __init__(self, cell, scalars=None):
        """
        given a Cell object, this class initializes an object that is easier to work with
        """
        self.scalars = ["membrane voltage"] if scalars is None else scalars
        self.n_time_points = len(self.scalars[0])
        self.line_pairs = []
        self.points = []
        self.diameters = []  # defined per point
        self.scalar_data = {name: [] for name in self.scalars}

        self.__parse_cell(cell)

    def __parse_cell(self, cell):
        """
        Given a cell object, this method parses the data to a format that lends itself easier to visualisation
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
        section_indices = []
        section_n_points = []
        for lv, sec in enumerate(cell.sections):  
            scalar_map = {"membrane voltage": sec.recVList
            }  # dictionary keeping track of which keyword belongs to which attribute of the cell section

            if sec.label in ['Soma', 'AIS', 'Myelin']:
                continue

            segs_limits = __construct_segment_limits(sec)

            current_seg = 0
            next_seg_flag = False 
            n_prev_points = int(np.sum(section_n_points))

            # append first point of the section
            self.points.append(sec.pts[0])
            self.diameters.append(sec.diamList[0])
            section_indices.append(lv)
            for i, pt in enumerate(sec.pts[1:]):  # append consecutive points plus connection
                self.points.append(pt)
                self.diameters.append(sec.diamList[i])
                section_indices.append(lv)
                index = n_prev_points + i
                self.line_pairs.append([index, index+1])
                current_seg, next_seg_flag = __update_segment_indices_(i, sec, segs_limits, next_seg_flag, current_seg)
                for scalar_data_name in self.scalars:  # write out scalar data
                    self.scalar_data[scalar_data_name].append(scalar_map[scalar_data_name][current_seg])
            section_n_points.append(len(sec.pts))

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

    def __write_vtk_frame(self, out_name, out_dir, time):
        """
        Writes out the current cell object 
        """
        def header_(out_name_=out_name):
            h=f"# vtk DataFile Version 4.0\n{out_name_}\nASCII\nDATASET POLYDATA\n"
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
                l += f"2 {p1} {p2}\n"
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
        with open(out_dir+"/"+out_name+"_{:06d}.vtk".format(time), "w+", encoding="utf-8") as of:
            of.write(header_(out_name))

            # Points
            of.write(f"POINTS {len(self.points)} float\n")
            of.write(points_str_(self.points))

            # Line
            of.write(f"LINES {len(self.line_pairs)} {3*len(self.line_pairs)}\n")  # LINES n_cells total_amount_integers
            # e.g. 2 16 765 means index 765 and 16 define a single line, the leading 2 defines the amount of points that define a line
            of.write(line_pairs_str_(self.line_pairs))

            # Diameters
            of.write(f"POINT_DATA {len(self.diameters)}\nSCALARS Diameter float 1\nLOOKUP_TABLE default\n")  # SCALARS name_of_data data_type n_components
            of.write(diameter_str_(self.diameters))

            # Scalar data (as of now only membrane voltages)
            for name, data in self.scalar_data.items():
                of.write(f"CELL_DATA {len(data)}\nSCALARS Vm float 1\nLOOKUP_TABLE default\n")
                of.write(scalar_data_str_(name))

    def write_vtk_frames(self, out_name, out_dir):
        for t in tqdm(range(self.n_time_points), desc="Writing vtk frames to {}".format(out_dir)):
                self.__write_vtk_frame(out_name, out_dir, time=t)

    def show_cell_3d(self, azim=-60, dist=10, elev=30, save=''):
        """
        Show the current cell in 3D using matplotlib
        """
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        
        for pair in self.line_pairs:
            pt1, pt2 = self.points[pair[0]], self.points[pair[1]]
            ax.plot3D(*np.array([pt1, pt2]).T, 'grey',lw=self.diameters[pair[0]]*1.5) 

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        
        ax.azim = azim
        ax.dist = dist
        ax.elev = elev
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        if save != '':
            plt.savefig(save)
        plt.show()

    def show_voltage_cell_3d(self, n_frame, azim=-60, dist=10, elev=30, save=''):
        """
        Plot the current CellViz object # TODO finish docstring
        """
        fig = plt.figure(figsize = (10,10))
        ax = plt.axes(projection='3d')
        
        for i, pt in enumerate(self.points):
            # Points within the same section
            x, y, z = pt
            d = self.diameters[i]    
            color = (self.scalar_data["membrane voltage"][n_frame] + 80)/100
            ax.plot3D([x[i],x[i+1]], [y[i],y[i+1]], [z[i],z[i+1]], color=mpl.cm.jet(color) ,lw=d[i]*1.5) 

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('off')
        
        ax.azim = azim
        ax.dist = dist
        ax.elev = elev
        ax.set_box_aspect([ub - lb for lb, ub in (getattr(ax, f'get_{a}lim')() for a in 'xyz')])
        if save != '':
            plt.savefig(save)
        plt.show()
