import pandas as pd
import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
from _decorators import *
from compatibility import multiprocessing_scheduler
from ..analyze._helper_functions import is_int
from _figure_array_converter import fig2np
from .. utils import convertible_to_int

def rasterplot2(st, ax = None, x_offset = 0, c = None, 
                    plot_kwargs = {}, y_offset = None, y_plot_length = 1):
    if ax is None:
        ax = plt.figure().add_subplot(111)
    if c is not None:
        plot_kwargs['c'] = c
    st = st[[c for c in st.columns if convertible_to_int(c)]]
    if y_offset is None:
        y = len(st)
    else:
        y = y_offset
    for i, v in st.iterrows():
        dummy = [([v-x_offset, v-x_offset], [y, y-y_plot_length]) for v in list(v)]
        for d in dummy:
            ax.plot(d[0], d[1], **plot_kwargs)
        y = y-1

def rasterplot2_pdf_grouped(pdf, grouplabel, ax = None, xlim = None, x_offset = 0, color = 'k'):
    if ax is None:
        fig = plt.figure(figsize = (7,4), dpi = 600)
        ax = fig.add_subplot(111)
    yticks = []
    ylabels = []
    offset = 0
    labels = pdf[grouplabel].drop_duplicates()
    for label in labels:
        df = pdf[pdf[grouplabel] == label]
        offset += len(df)
        rasterplot2(df, ax = ax, y_offset=offset, x_offset = x_offset,\
                          plot_kwargs = {'c': color, 'linewidth': 2, 'solid_capstyle': 'butt'})
        plt.axhline(offset, c = 'grey', linewidth = .1)
        yticks.append(offset - len(df) / 2.)
        ylabels.append(label)
    if xlim is not None: ax.set_xlim(*xlim)
    ax.set_ylim(0,offset + .2)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    try:
        import seaborn as sns
        sns.despine()
    except ImportError:
        pass
    try:
        from IPython import display
        display.display(fig)
    except (ImportError, UnboundLocalError):
        pass
    plt.close()    

@dask_to_pandas
@return_figure_or_axis
def rasterplot(df, colormap = None, fig = None, label = None, groupby_attribute = None, tlim = None, figsize = (15,3), reset_index = True):
    '''
    creates a rasterplot,
    expects dataframe in the usual spike times format
    
    if df is a dask.DataFrame: parallel plotting is used (not recommended, causes bad quality)
    if df is a pandas.DataFrame, serial plotting is used
    '''
    
    if reset_index:
        df = df.reset_index()
    
    if groupby_attribute:
        groups = df.groupby(groupby_attribute)
        for label, group_df in groups:
            rasterplot(group_df, colormap=colormap, fig=fig, label = label, groupby_attribute = None, reset_index = False)    
        return fig
    
    relevant_columns = [_ for _ in df.columns if is_int(_)]
    df = df[relevant_columns]
    times_all = []
    trails_all = []
    ax = fig #before: ax = fig.add_subplot(1,1,1), now managed by decorator  return_figure_or_axis
    
    #fig.patch.set_alpha(0.0)
    #axes = plt.axes()
    #axes.patch.set_alpha(0.0) 
        
    for index, row in df.iterrows():
        row = list(row)
        times = [time for time in row if is_int(time)]
        times_all.extend(times)
        trails_all.extend([index]*len(times))
    if colormap: 
        color =  colormap[label]
        ax.plot(times_all, trails_all, 'k|', color = color, label = label)
    else:
        ax.plot(times_all, trails_all, 'k|', label = label)
    if tlim:
        ax.set_xlim(tlim)
    #plt.gca().set_position([0.05, 0.05, 0.95, 0.95])    

    return fig
        

######################################
# currently not used: version supporting parralel plotting




# import pandas as pd
# import dask
# import dask.dataframe as dd
# import matplotlib.pyplot as plt
# from _decorators import *
# from ..settings import multiprocessing_scheduler
# from ..analyze._helper_functions import is_int
# from _figure_array_converter import fig2np
# 
# 
# @return_figure_or_axis
# def rasterplot(df, colormap = None, fig = None, label = None, groupby_attribute = None, tlim = None, figsize = (15,3)):
#     '''
#     creates a rasterplot,
#     expects dataframe in the usual spike times format
#     
#     if df is a dask.DataFrame: parallel plotting is used (not recommended, causes bad quality)
#     if df is a pandas.DataFrame, serial plotting is used
#     '''
#     df = df.reset_index()
#     
#     if isinstance(df, pd.DataFrame):
#         fun = lambda x: rasterplot_pd(x, colormap = colormap, fig = fig, label = label, groupby_attribute = groupby_attribute, tlim = tlim, figsize = figsize)
#         return fun(df)    
#     
#     elif isinstance(df, dd.DataFrame):
#         if tlim is None:
#             raise ValueError("For parallel execution, which is enabled by passing a dask dataframe, tlim has to be set like [0 300]")
#         
#         fun = lambda x: rasterplot_pd(x, colormap = colormap, fig = None, label = label, groupby_attribute = groupby_attribute, tlim = tlim, figsize = figsize)
#         fun2 = lambda x: pd.Series({'A': fig2np(fun(x))})
#         figures_list = df.map_partitions(fun2).compute(get = dask.async.get_sync)
#         
#         plt.axis('off')
#         #print figures_list
#         for lv, img in enumerate(figures_list.values):
#             ax = fig.imshow(img, interpolation='nearest') #be aware: decorator handles figure and allways passes axes as fig argument
#             
#         fig.set_position([0, 0, 1, 1])            
#         #plt.close(fig)
#         #print('asdasdasdasd: %s' % type(fig))
#         return fig
#     
#     
#     
# @return_figure_or_axis    
# def rasterplot_pd(df, colormap = None, fig = None, label = None, groupby_attribute = None, tlim = None, figsize = (15,3)):         
# 
#     if groupby_attribute:
#         groups = df.groupby(groupby_attribute)
#         for label, group_df in groups:
#             rasterplot_pd(group_df, colormap=colormap, fig=fig, label = label, groupby_attribute = None)    
#         return fig
#     
#     relevant_columns = [_ for _ in df if is_int(_)]
#     times_all = []
#     trails_all = []
#     ax = fig #fig.add_subplot(1,1,1)
#     
#     fig.patch.set_alpha(0.0)
#     axes = plt.axes()
#     axes.patch.set_alpha(0.0) 
#     
#         
#     for index, row in df.iterrows():
#         row = list(row)
#         times = [time for time in row if is_int(time)]
#         times_all.extend(times)
#         trails_all.extend([index]*len(times))
#     if colormap: 
#         color =  colormap[label]
#         ax.plot(times_all, trails_all, 'k|', color = color, label = label)
#     else:
#         ax.plot(times_all, trails_all, 'k|', label = label)
#     if tlim:
#         ax.set_xlim(tlim)
#     plt.gca().set_position([0.05, 0.05, 0.95, 0.95])    
#     try:
#         plt.close(fig)
#     except TypeError:
#         pass
#     
#     return fig
#         