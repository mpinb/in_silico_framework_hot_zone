import numpy as np
import matplotlib.pyplot as plt
import dask
import os
import glob


import IPython.display
import jinja2

def get_default_axis(range_var):
    if range_var == 'Vm':
        return (0,1600), (-80,0)
    if range_var == 'Ih.gIh':
        return (0,1600), (-2*10e-5,2*10e-5)
    if range_var == 'na_ion':
        return (0,1600), (0,300)
    if range_var == 'Ca_HVA.ica':
        return (0,1600), (-500*10e-5,2*10e-5)
    if range_var == 'Ca_LVAst.ica':
        return (0,1600), (-500*10e-5,2*10e-5)
    else:
        return (0,1600), (-2*10e-5,2*10e-5)
    
    
    
def display_animation(animID, files, interval=10, style=False):
    '''creates an IPython animation out of files specified in a globstring or a list of paths.
    
    animID: unique integer to identify the animation in the javascript environment of IPython
    files: globstring or list of paths
    interval: time interval between frames'''
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('/'))
    template = env.get_template('/nas1/Data_arco/animation_template.html')

    try:            # frames are local and can be expanded
        listFrames = sorted(glob.glob(files))
    except:         # frames are remote ressources
        listFrames = files
    htmlSrc = template.render(ID=animID, listFrames=listFrames, interval=interval, style=style)
    
    return IPython.display.HTML(htmlSrc)


def find_closest_index(list_, value):
    '''returns index of value within list_, which is closest to the value specified in the arguments'''
    m = min(range(len(list_)), key=lambda i: abs(list_[i]-value))
    return m

def get_synapse_points(cell, n):
    pass

def get_lines(cell, n, range_vars = 'Vm'):
    '''returns list of dictionaries of lines that can be display ed using the plot_lines function'''
    if isinstance(range_vars, str):
        range_vars = [range_vars]
        
    cmap = {'Soma': 'k', 'Dendrite': 'b', 'ApicalDendrite': 'r', 'AIS': 'r', 'Myelin': 'y'}
    out_all_lines = []
    for currentSec in cell.sections:
        out = {}
        currentSec_backup = currentSec
        #don't plot soma
        if currentSec.label == 'Soma':
            continue
        parentSec = currentSec.parent
        
        #compute distance from current section to soma
        dist = 0.0
        parentLabel = parentSec.label

        while parentLabel != 'Soma':
            dist += parentSec.L
            currentSec = parentSec
            parentSec = currentSec.parent
            parentLabel = parentSec.label
        
        #now calculate it segment wise.
        #First point is branchpoint of parent section, because otherwise there will be a gap in the plot
        distance_dummy = [dist - (1-list(currentSec_backup.parent)[-1].x)*currentSec_backup.parent.L]
        #calculate each segment distance
        for seg in currentSec_backup:
            distance_dummy.append(dist + seg.x*currentSec_backup.L)
        
        # voltage traces are a spiecal case
        if range_vars[0] == 'Vm':
            traces_dummy = [currentSec_backup.parent.recVList[-1][n]]
            for vec in currentSec_backup.recVList:
                traces_dummy.append(vec[n])       
        # other range vars are saved differently in the cell object compared to Vm       
        else:
            vec_list = currentSec_backup.recordVars[range_vars[0]]
            try:
                traces_dummy = [currentSec_backup.parent.recordVars[range_vars[0]][-1][n]]
            except:
                [np.NaN]
            if not vec_list: continue #if range mechanism is not in section: continue
            for vec in vec_list:
                traces_dummy.append(vec[n])
                #sec.recordVars[range_vars[0]][lv_for_record_vars]
        out['x'] = distance_dummy
        out['y'] = traces_dummy
        out['color'] = cmap[currentSec_backup.label]
        out['label'] = currentSec_backup.label
        out['t'] = cell.t[n]
        out_all_lines.append(out)
    return out_all_lines
#%time silent = [get_lines(cell, i) for i in range(1000)]

def init_fig(xlim = (0,1500), ylim = (-80,0)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig, ax

def plot_lines(lines, ax):
    '''generate plots out of lines '''
    for line in lines:
        x = line['x']
        y = line['y']
        t = line['t']
        del line['x']
        del line['y']
        del line['t']
        ax.plot(x,y,**line)
        ax.set_title("%.3f" % t)


@dask.delayed
def _in_parallel_context(path, lines_object, xlim = (0,1500), ylim = (-80,0)):
    '''helper function to launch generation of images in parallel'''
    fig, ax = init_fig(xlim, ylim)
    plot_lines(lines_object, ax)
    fig.savefig(path)
    plt.close()
    
def parallelMovieMaker(basedir, lines, xlim = (0,1500), ylim = (-80,0), return_glob = False):
    '''creates figures in parallel and returns animation object.
    basedir: path to store images
    lines: list of dictionaries of lines (generated by function generate_lines)
    xlim: limits of x axis
    ylim: limits of y axis
    return_glob: return globstring to image files
    '''
    import tempfile
    basepath = tempfile.mkdtemp(dir = basedir, prefix = 'animation_')
    print "files are in folder %s " % basepath
    paths = [os.path.join(basepath, str(i).zfill(6) + '.png') for i in range(len(lines))]
    
    delayed_list = [_in_parallel_context(path, line, xlim = xlim, ylim = ylim) for  path, line in zip(paths, lines)]
    dask.delayed(delayed_list).compute(get = dask.multiprocessing.get)
    
    ani = display_animation(np.random.randint(10000000000000), paths)
    if return_glob:
        return ani, os.path.join(basepath, '*.png')
    else: 
        return ani
    
    
def cell_to_ipython_animation(cell, xlim = None, ylim = None, tstart = 245, tend = 310, tstep = 1, \
                              range_vars = 'Vm', return_glob = False, plot_synaptic_input = False):
    '''takes a cell object and creates a 2d animation
    
    xlim: tuple like (0, 1600) that defines the limits of the x axis.
            If None, tries to get default values for range var from get_default_axis
            Default: None. 
    ylim: tuple like (-80 0) that defines the limits of the y axis
            If None, tries to get default values for range var from get_default_axis
            Default: None.     
    tstart: time when animation should be started
    tend: time when animation should be stopped
    tstep: timestep between frames
    to_folder: folder, where animation should be saved
    range_vars: str or list of str: range vars to record, default: 'Vm'
    return glob: if false: only returns animation object, else returns 
                tuple with globstring and animation object'''
    if isinstance(range_vars, str): range_vars = [range_vars]

    index_start = find_closest_index(cell.t, tstart)
    index_step = find_closest_index(cell.t, tstep) + 1
    nframes = int(find_closest_index(cell.t, tend - tstart) / float(index_step))
    index_stop = index_start + (nframes - 1) * index_step
    print "Starting at index %s, which corresponds to time %s ms" % (str(index_start), str(cell.t[index_start]))
    print "Index stepsize is %s, which corresponds to %s ms" % (str(index_step), str(cell.t[index_step]))
    print "Stop index is %s, which corresponds to %s ms" % (str(index_stop), str(cell.t[index_stop]))
    print "Number of frames is %s" % nframes
    
    indices = np.arange(index_start, index_stop + index_step, index_step)
    lines = [get_lines(cell, index, range_vars = range_vars) for index in indices]
    
    if xlim is None and ylim is None:
        xlim, ylim = get_default_axis(range_vars[0])
    
    return parallelMovieMaker('asdasdasdasd', lines, xlim = xlim, ylim = ylim, return_glob = return_glob)