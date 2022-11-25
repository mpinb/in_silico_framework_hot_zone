import Interface as I
from helper_functions import get_neuron_param_file, get_main_bifurcation_section, get_section_distances_df, get_spatial_bin_names

class WeightPlot():
    def __init__(self, mdb):
        """
        Given a model database, this class initializes a bunch of variables to be able to plot the weight plots, such as:
        - neuron parameter file
        - cell
        - section distances dataframe, spatial bin names, sections
        """
        self.neuron_param_file = get_neuron_param_file(mdb) # .keys()
        self.neup = I.scp.NTParameterSet(self.neuron_param_file)
        self.cell = I.scp.create_cell(self.neup.neuron)
        self.section_distances_df = get_section_distances_df(self.neup)
        self.spatial_bin_names = get_spatial_bin_names(self.section_distances_df)
        self.spatial_bin_names_df = I.pd.DataFrame(index = get_spatial_bin_names(self.section_distances_df))
        self.sorted_index=[]

        self.tuft_sections = []
        self.oblique_sections = []
        self.trunk_sections = []

        self.__parse_sections()
    
    def __parse_sections(self):
        main_bifurc_sec = get_main_bifurcation_section(self.cell)
        self.trunk_sections = [main_bifurc_sec]
        while True:
            sec = self.trunk_sections[-1].parent
            if  sec.label == 'Soma':
                break
            else:
                self.trunk_sections.append(sec)
        self.tuft_sections = []
        self.oblique_sections = []
        for sec in self.cell.sections:
            if not sec.label == 'ApicalDendrite':
                continue
            secp = sec.parent
            while True:
                if secp.label == 'Soma':
                    if not sec in self.trunk_sections:
                        self.oblique_sections.append(sec)
                    break
                if secp == main_bifurc_sec:
                    self.tuft_sections.append(sec)
                    break
                secp = secp.parent
        self.spatial_bin_names_df['soma_dist'] = I.np.nan
        for lv, sec in enumerate(self.cell.sections):
            if not sec.label in ['Dendrite', 'ApicalDendrite']:
                continue
            n_bins = self.section_distances_df.loc[lv]['n_bins']
            bin_of_each_point = I.np.digitize(sec.relPts, I.np.linspace(0,1,n_bins + 1))
            for bin_ in range(1,n_bins+1):
                xs = [x for x, b in zip(sec.relPts, bin_of_each_point) if b == bin_]
                mean_x = I.np.mean(xs)
                bin_name = str(lv) + '/' + str(bin_)
                self.spatial_bin_names_df.loc[bin_name,'soma_dist'] = self.cell.distance_to_soma(sec, mean_x)
                self.spatial_bin_names_df.loc[bin_name,'label'] = '3_tuft' if sec in self.tuft_sections\
                                                                else '2_trunk' if sec in self.trunk_sections\
                                                                else '1_oblique' if sec in self.oblique_sections\
                                                                else '0_basal'
        self.spatial_bin_names_df['index'] = range(len(self.spatial_bin_names_df))
        self.sorted_index = [0] + list(self.spatial_bin_names_df.sort_values(['label','soma_dist']).dropna()['index'])  #.values

    def plot_weights(self, weight, n_celltypes, n_spatial_bins, temporal_window_width, weight_scaling, celltype,
    AUCs, train_loss, test_loss, categories=True, t=0, return_axes=True, i=None, ts=None):
        """
        Given the weights of a neural network, this class method makes a visualisation of the neuron, the weights per section,
        and the train and test loss.

        Arguments:
            weight: array of network weights to be visualised. this matrix will be reshapes according to the dimensions:
                n_celltypes: amount of celltypes
                n_spatial_bins: amount of spatial bins
                n_temporal_window_width: width of temporal window
            celltype: 0 for excitatory, 1 for inhibitory (I think?)
            AUCs: (array) AUC scores during training
            train_loss: (array) train losses during training
            test_loss: (array) test losses of cross-validation during training
            categories: (bool) label the different sections of the neuron
            i: index of best epoch
            ts: timescale
        """
        def show_cell(ax, weights=None, scaling=None, t=0, show_bins=False):
            for lv, sec in enumerate(self.cell.sections):
                if sec.label in ['Soma', 'AIS', 'Myelin']:
                    continue
                n_bins = self.section_distances_df.loc[lv]['n_bins']
                bin_of_each_point = I.np.digitize(sec.relPts, I.np.linspace(0,1,n_bins + 1))
                for bin_ in range(1,n_bins+1):
                    if bin_ == 1:
                        x = [sec.pts[0][0]]
                        z = [sec.pts[0][2]]
                    else:
                        x = [x[-1]]
                        z = [z[-1]]
                    
                    x = [x[0] for x, b in zip(sec.pts, bin_of_each_point) if b == bin_]
                    z = [x[2] for x, b in zip(sec.pts, bin_of_each_point) if b == bin_]
                    x = x[1:-1]
                    z = z[1:-1]
                    spatial_bin_name = str(lv) + '/' + str(bin_)
                    spatial_bin_index = self.spatial_bin_names.index(spatial_bin_name)
                    if weights is None:
                        color = 'grey'
                        if sec in self.trunk_sections:
                            color = 'red'
                        if sec in self.tuft_sections:
                            color = 'blue'
                        if sec in self.oblique_sections:
                            color = 'green'
                    else:
                        spatial_bin_name = str(lv) + '/' + str(bin_)
                        spatial_bin_index = self.spatial_bin_names.index(spatial_bin_name)
                        color = get_color(weights[spatial_bin_index, t], -scaling, scaling)
                    if show_bins == True:
                        ax.plot(x,z)
                    else: 
                        ax.plot(x,z, color = color)

        ### figure specifications
        axscale = 1.5
        fig, axes = I.plt.subplots(2, 4, 
                                gridspec_kw={'width_ratios': [20,70, 1,20]}, 
                                figsize = (9*axscale,4*axscale), dpi = 150)
        ### preprocess weights
        weights1 = weight.reshape(n_celltypes, n_spatial_bins, temporal_window_width)[:,self.sorted_index,:]
        scaling = max(weights1.max(),-weights1.min())

        ### show weights
        for lv in range(n_celltypes):
            ax = axes[lv,1]
            im = ax.imshow(I.np.transpose(weights1[lv])*weight_scaling, vmin=-scaling, vmax=scaling, 
                            cmap = 'seismic', interpolation = 'none')
        ### colorbar
        gs = ax.get_gridspec()
        # remove the underlying axes
        for ax in axes[:, 2]:
            ax.remove()
        axbig = fig.add_subplot(gs[:, 2])
        fig.colorbar(im, cax = axbig) 
        axbig.yaxis.set_ticks_position('left')
        #fig.colorbar(im, cax = axes[1,0])
        ### tickss (names and position)
        separators = self.spatial_bin_names_df.label.value_counts().sort_index().cumsum()
        axbig.set_aspect('auto', anchor = 'W')
        tick_posititions = []
        _ = 0
        for s in separators:
            tick_posititions.append((s+_)/2)
            _ = s

        tick_names = ['basal','oblique','trunk','tuft']

        for lv in range(n_celltypes):
            ax = axes[lv,1]
            ax.set_xticks(tick_posititions)
            if lv < n_celltypes-1:
                ax.set_xticklabels('')
            else:
                ax.set_xticklabels(tick_names)

            for lv in separators:
                ax.axvline(lv+0.5, color = 'k', linestyle = '--', linewidth = 1)

            ax.set_yticks([0,20,40,60,80])
            ax.set_yticklabels([-80,-60,-40,-20,'now'])

        axes[0,1].set_ylabel('EXC\ninput history / ms')
        axes[1,1].set_ylabel('INH\ninput history / ms') 
        axes[0,1].set_aspect('equal', anchor ='SE')
        axes[1,1].set_aspect('equal', anchor ='NE')

        #axes[0,1].set_title('EXC')
        #axes[1,1].set_title('INH')
        
        I.plt.tight_layout()
        I.sns.despine()

        gs = ax.get_gridspec()
        # remove the underlying axes
        for ax in axes[:, 0]:
            ax.remove()
        axcell = fig.add_subplot(gs[:, 0])

        ax = axcell
        
        if not categories:
            show_cell(ax, weights = weight.reshape(n_celltypes, n_spatial_bins, temporal_window_width)[celltype]*weight_scaling,
                scaling = scaling, t=t)
            #axes[0,1].axhline(t)

        else:
            show_cell(ax)
        
        I.sns.despine(ax = ax, left = True, bottom = True)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        axes[1,1].get_xticklabels()[0].set_color("grey")
        axes[1,1].get_xticklabels()[1].set_color("green")
        axes[1,1].get_xticklabels()[2].set_color("red")
        axes[1,1].get_xticklabels()[3].set_color("blue")
        
        # plot AUCs
        ax = axes[0,3]
        ax.plot(AUCs)
        ax.set_ylim(0.94,0.98)
        ax.set_ylabel('AUROC')

        # plot loss
        ax = axes[1,3]
        ax.set_ylabel('loss')    
        ax.plot(train_loss, label = 'train')    
        ax.plot(test_loss, label = 'test')
        ax.set_xlabel('epoch')
        ax.legend()

        if ts is not None:
            title = 'history: {} ms'.format(-ts[t]-1)
            axes[0,1].set_title(title)
        if i is not None:
            axes[0,3].axvline(i-1, color = 'k', linestyle = '--', linewidth = 1)
            axes[1,3].axvline(i-1, color = 'k', linestyle = '--', linewidth = 1)
        axes[celltype,1].axhline(t, color = 'k', linestyle = '--', linewidth = 1)

        if return_axes:
            return axes


def get_color(x, vmin, vmax, cm = I.matplotlib.cm.seismic):
    color_position = (x-vmin)/(vmax-vmin)
    return cm(color_position)
