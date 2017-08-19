from _decorators import *

@dask_to_pandas
@return_figure_or_axis # has to be before subsequent_calls_per_line
@subsequent_calls_per_line
def histogram(hist_bins, 
                         colormap = None,
                         fig = None,
                         label = None,
                         groupby_attribute = None):
    
    '''expects a tuple of the format (bins,hist)
    
    Supports groups: simply pass a Series of the format.
    labelA: (bins,hist)
    labelB: (bins,hist)
    In this case, the label attribute has no function (to be precise: it is overwritten
    by the decorator subsequent_calls_per_line)
    '''
    if isinstance(hist_bins, pd.Series):
        hist_bins = hist_bins.iloc[0]
        
    bins = hist_bins[0]
    hist = hist_bins[1]
    #add points, so stepfunction allways starts and ends on the x-axis
    x = list(bins) + [bins[-1]]
    y = [0] + list(hist) + [0]
    
    ax = fig#.add_subplot(1,1,1) automatically done by return_figure_or_axis
    if colormap:
        color =  colormap[label]
        ax.step(x,y, color = color, label = label)
    else:
        ax.step(x,y, label = label)
    
    try:
        plt.close(fig)
    except TypeError:
        pass    
    
    return fig
    