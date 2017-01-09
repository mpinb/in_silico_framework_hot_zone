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
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('/'))
    template = env.get_template('/nas1/Data_arco/animation_template.html')

    try:            # frames are local and can be expanded
        listFrames = sorted(glob.glob(files))
    except:         # frames are remote ressources
        listFrames = files
    htmlSrc = template.render(ID=animID, listFrames=listFrames, interval=interval, style=style)
    
    return IPython.display.HTML(htmlSrc)


def find_closest_index(list_, value):
    m = min(range(len(list_)), key=lambda i: abs(list_[i]-value))
    return m

def get_lines(cell, n, range_vars = 'Vm'):
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
    fig, ax = init_fig(xlim, ylim)
    plot_lines(lines_object, ax)
    fig.savefig(path)
    plt.close()
    
def parallelMovieMaker(basedir, lines, xlim = (0,1500), ylim = (-80,0), return_glob = False):
    import tempfile
    basepath = tempfile.mkdtemp(dir = basedir, prefix = 'animation_')
    print "files are in folder %s " % basepath
    paths = [os.path.join(basepath, str(i).zfill(6) + '.png') for i in range(len(lines))]
    
    delayed_list = [_in_parallel_context(path, line, xlim = xlim, ylim = ylim) for  path, line in zip(paths, lines)]
    
    dask.compute(*delayed_list, get = dask.multiprocessing.get)
    
    ani = display_animation(np.random.randint(10000000000000), paths)
    if return_glob:
        return ani, os.path.join(basepath, '*.png')
    else: 
        return ani
    
    
def cell_to_ipython_animation(cell, variable = '', xlim = None, ylim = None, tstart = 245, tend = 310, tstep = 1, \
                              to_folder = None, range_vars = 'Vm', return_glob = False):
    if isinstance(range_vars, str): range_vars = [range_vars]
    print range_vars

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