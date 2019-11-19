import Interface as I

######################
# functions to extract branchpoints
######################

def connected_to_dend_beyond(cell, sec, beyond_dist, n_children_required = 2):
    '''returns true if at least two children of the branchpoint reach beyond dist'''
    if cell.distance_to_soma(sec, 1) > beyond_dist: # and sec.label in ('ApicalDendrite', 'Dendrite'):
        return True
    else:
        dummy = sum(connected_to_dend_beyond(cell, c, beyond_dist, n_children_required = 1) 
                        for c in sec.children()
                        # if sec.label in ('ApicalDendrite', 'Dendrite')
                   )
        if dummy >= n_children_required:
            return True
        else:
            return False
        

def get_inner_sec_dist_list(cell, select = ['ApicalDendrite', 'Dendrite']):
#    sec_dist_dict = {cell.distance_to_soma(sec, 1.0): sec 
    sec_dist_dict = {sec.pts[-1][2] - 706: sec 
                 for sec in cell.sections
                 if connected_to_dend_beyond(cell, sec, 1000)
                 and sec.label in select
                }
    return sec_dist_dict


def get_branching_depth(cell, sec):
    depth = connected_to_dend_beyond(cell, sec, 1000)
    if sec.parent.label.lower() == 'soma':
        return depth
    else:
        return depth + get_branching_depth(cell, sec.parent)

def get_branching_depth_series(cell):
    '''
    Careful: z-depth only accurate for D2-registered cells!
    
    returns a series that contains the pia distance as index and  
    a tuple (biforcation order, section) as value'''
    
    inner_sections = get_inner_sec_dist_list(cell)
    inner_sections_branching_depth = {k: (get_branching_depth(cell, sec), sec)
                                      for k, sec in inner_sections.iteritems()}
    inner_sections_branching_depth = I.pd.Series(inner_sections_branching_depth)
    return inner_sections_branching_depth

def get_main_bifurcation_section(cell):
    sec_dist_list = get_branching_depth_series(cell)
    sec_dist_list_filtered = [sec[1] for sec in sec_dist_list if sec[0] == 1]
    assert(len(sec_dist_list_filtered) == 1)
    return sec_dist_list_filtered[0]


    

#######################################
# functions to read hoc files
#######################################
def get_cell_object_from_hoc(hocpath):
    '''returns cell object, which allows accessing points of individual branches'''    
    # import singlecell_input_mapper.singlecell_input_mapper.cell
    # ssm = singlecell_input_mapper.singlecell_input_mapper.cell.CellParser(hocpath)    
    # ssm.spatialgraph_to_cell()
    # return ssm.cell
    neuron_param = {'filename': hocpath}
    neuron_param = I.scp.NTParameterSet(neuron_param)
    cell = I.scp.create_cell(neuron_param)
    return cell

##########################################
# database containing mikes cells
##########################################
# id: '2019-09-09_19894_YfrHaoO'
mdb_cells = I.ModelDataBase('/nas1/Data_arco/results/20190909_VPM_contact_analysis_for_mike_review') 

###########################################
# test, whether two ranges overlap
#############################################
def range_overlap(beginning1, end1, beginning2, end2, bool_ = True):
    out = min(end1, end2) - max(beginning1, beginning2)
    out = max(0,out)
    if bool_:
        out = bool(out)
    return out

assert(not range_overlap(0,1,1,2))
assert(not range_overlap(0,1,1.1,2))
assert(not range_overlap(0,1,2,3))
assert(range_overlap(0,1,0.9,2))
assert(range_overlap(0,1.1,1,2))
assert(range_overlap(0,1.1,0.9,1.1))
assert(range_overlap(0,1.1,0.9,1))
assert(range_overlap(0,1.1,-1,2))
assert(range_overlap(0,1,0,1))

#########################################
# class that can visualize the synapse counts and can 
# compute the density profiles
#########################################
def get_dist(x1, x2):
    assert(len(x1) == len(x2))
    return I.np.sqrt(sum((xx1-xx2)**2 for xx1, xx2 in zip(x1, x2)))

def get_L_from_sec(sec, lv):
    out = 0
    for lv in range(1, lv + 1):
        assert(lv-1) >= 0
        out += get_dist(sec.pts[lv-1], sec.pts[lv])
    return out

class Dendrogram:
    def __init__(self, cell, title = '', colormap_synapses = None):
        # cell object needs to implement the following functionality:
        #     cell.soma
        #     section.children()
        #     section.L
        self.cell = cell
        self.title = title
        self.dendrogram_db = []
        self.dendrogram_db_by_name = {}
        if colormap_synapses:
            self.colormap_synapses = colormap_synapses
        else: 
            self.colormap_synapses = I.defaultdict(lambda: 'k')
        self._cell_to_dendrogram(self.cell.soma)
        self._soma_to_dendrogram(self.cell)
        self._set_initial_x()
        self._compute_main_bifurcation_section()
    
    def _add_synapse(self, l, label, x):
        if not 'synapses' in l:
            l['synapses'] = {}
        if not label in l['synapses']:
            l['synapses'][label] = []
        l['synapses'][label].append(x)
        
    def add_synapses(self, label, synlist, mode = 'from_anatomical_synapse_mapper'):
        if mode == 'from_anatomical_synapse_mapper':
            for syn in synlist:
                (sec_lv, seg_lv), (dist, x, point) = syn
                sec = self.cell.sections[sec_lv]
                l = self.get_db_by_sec(sec)
                x = get_L_from_sec(sec, seg_lv) + x + l['x_offset'] + l['x_dist_start']
                self._add_synapse(l, label, x)              
        elif mode == 'id_absolute_position':
            for syn in synlist:
                sec_lv, x = syn
                sec = self.cell.sections[sec_lv]
                l = self.get_db_by_sec(sec)
                x = x + l['x_offset'] + l['x_dist_start']
                self._add_synapse(l, label, x)   
        elif mode == 'id_relative_position':
            for syn in synlist:
                sec_lv, x = syn
                sec = self.cell.sections[sec_lv]
                l = self.get_db_by_sec(sec)
                x = x*sec.L + l['x_offset'] + l['x_dist_start']
                self._add_synapse(l, label, x)   
                           
    def _cell_to_dendrogram(self, basesec = None, basename = '', x_dist_base = 0):
        if basename:
            basename = basename.split('__')
        else:
            basename = ('', '', '')
        for lv, sec in enumerate(sorted(basesec.children(), key = lambda x: x.label)):
            if sec.label not in ('Dendrite', 'ApicalDendrite', 'Soma'):
                continue
        
            x_offset = get_dist(basesec.pts[-1], sec.pts[0])
            x_dist_start = x_dist_base 
            x_dist_end = x_dist_start + x_offset + sec.L
            
            name = sec.label + '__' + (basename[1] + '_' if basename[1] else '') + str(lv) + '__' + str(I.np.round(sec.L))
            self.dendrogram_db.append({'name': name, 'x_dist_start': x_dist_start, 'x_dist_end': x_dist_end, 
                                       'sec': sec, 'x_offset': x_offset})
            
            self._cell_to_dendrogram(sec, name, x_dist_end)
        self.dendrogram_db_by_name = {d_['name']: d_  for d_ in self.dendrogram_db}

    def _soma_to_dendrogram(self, cell):
        soma_sections = [s for s in cell.sections if s.label.lower() == 'soma']
        print "adding {} soma sections".format(len(soma_sections))
        for lv, sec in enumerate(soma_sections):
            self.dendrogram_db.append({'name': 'Soma__{}__0'.format(lv), 'x_dist_start': I.np.NaN, 'x_dist_end': I.np.NaN, 
                                       'sec': sec, 'x_offset': I.np.NaN})
            
        
    def _set_initial_x(self):
        for lv, l in enumerate(sorted(self.dendrogram_db, key = lambda x: x['name'].split('__')[1])):
            l['x'] = lv
    
    def plot(self, fig = None):
        if fig is None:
            fig = I.plt.figure(figsize = (8,10), dpi = 200)
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(413)
        ax3 = fig.add_subplot(414)

        xlim = self._plot_dendrogram(ax)
        self._plot_dendrite_and_synapse_counts(ax2, xlim)
        self._plot_synapse_density(ax3, xlim)
        return fig
        
    def _plot_dendrogram(self, ax = None, colormap = {'Dendrite': 'grey', 'ApicalDendrite': 'r'}):
        # uses: self.colormap_synapses
        if ax is None:
            fig = I.plt.figure(figsize = (8,10), dpi = 200)
            ax = fig.add_subplot(211)
            
        for lv, l in enumerate(self.dendrogram_db):
            label, tree, L = l['name'].split('__')
            if not label in ['Dendrite', 'ApicalDendrite']:
                continue
            x = l['x']
            parent = self.get_parent_by_name(l['name'])
            if parent is None:
                parent_x = 0
            else:
                parent_x = parent['x']
            ax.plot([l['x_dist_start'], l['x_dist_start'] + l['x_offset']], [x, x], ':', c = 'grey', linewidth = .5)                
            ax.plot([l['x_dist_start'] + + l['x_offset'], l['x_dist_end']], [x, x], c = colormap[label], linewidth = .5)
            
            ax.plot([l['x_dist_start'], l['x_dist_start']], [x, parent_x], c = colormap[label], linewidth = .5)
            if 'synapses' in l:
                for syntype in l['synapses']:
                        for syn in l['synapses'][syntype]:
                            ax.plot([syn, syn], [x+.5, x-.5], c = self.colormap_synapses[syntype], linewidth = .3)
                    
            if 'main_bifurcation' in l:
                ax.plot([l['x_dist_end']],[l['x']], 'o')
        ax.set_title(self.title)
        ax.set_ylabel('branch id')
        
        xlim = ax.get_xlim()
        return xlim
    
    def _compute_dendrite_and_synapse_counts(self, dist_end = None, binsize = 50):
        if dist_end is None:
            dist_end = self._get_max_somadistance()
        bins = I.np.arange(0, dist_end + binsize, binsize)
        
        dendrite_density = [self.get_amount_of_dendrite_in_bin(bins[lv], bins[lv+1]) for lv in range(len(bins) - 1)]
        synapse_density = [self.get_number_of_synapses_in_bin(bins[lv], bins[lv+1]) for lv in range(len(bins) - 1)]
        synapse_density_apical = [self.get_number_of_synapses_in_bin(bins[lv], bins[lv+1], select = ['ApicalDendrite']) 
                                  for lv in range(len(bins) - 1)]
        synapse_density_basal =  [self.get_number_of_synapses_in_bin(bins[lv], bins[lv+1], select = ['Dendrite']) 
                                  for lv in range(len(bins) - 1)]
        dendrite_density_apical = [self.get_amount_of_dendrite_in_bin(bins[lv], bins[lv+1], select = ['ApicalDendrite']) 
                                   for lv in range(len(bins) - 1)]
        dendrite_density_basal =  [self.get_amount_of_dendrite_in_bin(bins[lv], bins[lv+1], select = ['Dendrite']) 
                                   for lv in range(len(bins) - 1)]       
                
        self.bins = bins
        self.dendrite_density =  dendrite_density = I.np.array(dendrite_density)
        self.synapse_density = synapse_density = I.np.array(synapse_density)
        self.synapse_density_apical = synapse_density_apical = I.np.array(synapse_density_apical)
        self.synapse_density_basal = synapse_density_basal = I.np.array(synapse_density_basal)
        self.dendrite_density_apical = dendrite_density_apical = I.np.array(dendrite_density_apical)
        self.dendrite_density_basal = dendrite_density_basal = I.np.array(dendrite_density_basal)  
        
    def _plot_dendrite_and_synapse_counts(self, ax, xlim, binsize = 50):
        self._compute_dendrite_and_synapse_counts(xlim[1], binsize = binsize)
        I.histogram((self.bins, self.dendrite_density), label = 'dendrite', colormap = {'dendrite': 'k'}, fig = ax)
        ax2 = ax.twinx()
        I.histogram((self.bins, self.synapse_density), label = 'dendrite', colormap = {'dendrite': 'r'}, fig = ax2)
        ax.set_ylabel('dendritic length / bin', color = 'k')
        ax2.set_ylabel('# synapses / bin', color = 'r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_xlim(xlim)
        
    def _plot_synapse_density(self, ax, xlim, binsize = 50):
        self._compute_dendrite_and_synapse_counts(xlim[1], binsize = binsize)
        I.histogram((self.bins, self.synapse_density / self.dendrite_density), label = 'total', colormap = {'total': 'k'}, fig = ax)
        # I.histogram((bins, synapse_density_basal / dendrite_density_basal), label = 'basal', 
        #             colormap = {'basal': 'grey'}, fig = ax)
        # I.histogram((bins, synapse_density_apical / dendrite_density_apical), label = 'apical', 
        #             colormap = {'apical': 'red'}, fig = ax)
        ax.legend()
        ax.set_xlim(xlim)
        ax.set_ylabel('# syn / micron dendritic length', color = 'k')
        ax.set_xlabel('somadistance / micron')
                
            
    def get_parent_by_name(self, name):
        sec = self.dendrogram_db_by_name[name]['sec'].parent
        if sec.label == 'Soma':
            return None
        else:
            return self.get_db_by_sec(sec) # next(l for l in self.dendrogram_db if l['sec'] == sec)
        
    def get_parent_dist_by_name(self, name):
        parent = self.get_parent_by_name(name)
        x = self.dendrogram_db_by_name[name]['x']
        if parent is None:
            return x
        else:
            return x-parent['x']
        
    def get_db_by_sec(self, sec):
        return next(l for l in self.dendrogram_db if l['sec'] == sec)
    
    def get_amount_of_dendrite_in_bin(self, min_, max_, select = ['Dendrite', 'ApicalDendrite']):
        
        out = 0
        for l in self.dendrogram_db:
            if not l['sec'].label in select:
                continue
            out += range_overlap(min_, max_, l['x_dist_start'], l['x_dist_end'], bool_ = False)
        # if out < 50: return 0
        return out
    
    def get_number_of_synapses_in_bin(self, min_, max_, select = ['Dendrite', 'ApicalDendrite'], label = 'VPMmike'):
        out = 0
        for l in self.dendrogram_db:
            if not 'synapses' in l:
                continue
            if not l['sec'].label in select:
                continue                
            for syn in l['synapses'][label]:
                if min_ <= syn < max_:
                    out+=1
        return out    
    
    def _compute_main_bifurcation_section(self):
        sec = get_main_bifurcation_section(self.cell)
        l = self.get_db_by_sec(sec)
        l['main_bifurcation'] = True
        self.hot_zone_dist = l['x_dist_end']
        
    def _get_max_somadistance(self):
        out = 0
        for l in self.dendrogram_db:
            out = max(out, l['x_dist_end'])
        self.max_somadist = out
        return out

#######################################
# class for exporting dendrogram objects
#######################################
class Synapse:
    '''mock synapse class that allows to use scp.writer.write_cell_synapse_locations'''
    def __init__(self, preCellType = None, secID = None, x = None):
        self.secID = secID
        self.x = x
        self.preCellType = preCellType

class ExportSynapses:
    def __init__(self, dendrogram):
        self.d = dendrogram
        self.synapses = None
        self.functionalMap = None
        self._create_synapses()
        
    def _create_synapses(self):
        d = self.d
        out = {}
        for l in d.dendrogram_db:
            sec_id = d.cell.sections.index(l['sec'])
            if 'synapses' in l:
                for syntype in l['synapses'].keys():
                    if not syntype in out:
                        out[syntype] = []
                    list_ = l['synapses'][syntype]
                    for syn in list_:
                        syn = syn - l['x_offset'] - l['x_dist_start']
                        syn = syn / l['sec'].L
                        out[syntype].append(Synapse(syntype, sec_id, syn))
        for k in out:
            out[k] = sorted(out[k], key = lambda x: (x.secID, x.x))
        self.synapses = out

    def create_functional_map_with_one_cell_per_synapse_from_cell_object(self):
        synapses_dict = self.synapses
        self.functionalMap = [(k, lv, lv) for k in synapses_dict.keys() for lv in range(len(synapses_dict[k]))]  
        
    def save(self, outdir):
        I.scp.writer.write_cell_synapse_locations(I.os.path.join(outdir, 'con.syn'), 
                                                  self.synapses, 
                                                  cellID='__no__anatomical__id')
        I.scp.writer.write_functional_realization_map(I.os.path.join(outdir, 'con.con'), 
                                                      self.functionalMap, 
                                                      '__no__anatomical__id')
        
# reload(hot_zone)
# k = 'CDK85'
# with I.silence_stdout:
#     cell = I.scp.create_cell(cell_params[k].neuron)
# d = hot_zone.Dendrogram(cell)   
# path = mdb_formalized_input['network_embedding'][k]['0'].join('con.syn')
# syndict = I.scp.read_synapse_realization(path)
# for k, v in syndict.iteritems():
#     print k
#     d.add_synapses(k, v, mode = 'id_relative_position')
# exp = hot_zone.ExportSynapses(d)
# exp.create_functional_map_with_one_cell_per_synapse_from_cell_object()
# exp.save('/nas1/Data_arco/')    
# syndict2 = I.scp.read_synapse_realization('/nas1/Data_arco/con.syn')
        
#####################################
# map synapses by profile
#####################################

def normalize_bins(bins, bifur, max_dist):
    '''normalizes bins such that soma is at 0, hot zone is at 1 and most distral branch is at 2'''
    out = []
    for b in bins:
        if b < bifur:
            out.append(float(b) / bifur)
        else:
            out.append((float(b) - bifur) / (max_dist - bifur) + 1)
    return out

def unnormalize_bins(bins, bifur, max_dist):
    '''reverses the normalization'''
    out = []
    for b in bins:
        if b < 1:
            out.append(b*bifur)
        else:
            out.append((b-1) * (max_dist-bifur) + bifur)
    return out

bins = I.np.random.rand(10) * 10
bins2 = normalize_bins(bins, 6, 11)
bins3 = unnormalize_bins(bins2, 6, 11)
I.np.testing.assert_almost_equal(bins, bins3)


class MapSynapsesByProfile:
    def __init__(self, dendrogram):
        self.d = dendrogram
        self.profiles = {}
    
    def _get_sections_and_lengths_in_interval(self, min_, max_, select = ['Dendrite', 'ApicalDendrite']):
        '''returns sections that can be found in a somadistance between min_ and max_.

        returns a 3-tuple, consisting of:
         - out_sec: list of PySection objects that can be found in a somadistance between min_ and max_
         - out_len: list containing the length of each section between min_ and max_
         - out_x_offset: some sections start proximal from min_. offset is the length from the beginning of
             the section till it reaches into to somadistance slice defined by min_ and max_'''
        
        d = self.d
        
        out_sec = []
        out_len = []
        out_x_offset = []
        for l in d.dendrogram_db:
            sec = l['sec']
            if not sec.label in select:
                continue        
            len_ = range_overlap(min_, max_, l['x_dist_start'] + l['x_offset'], l['x_dist_end'], bool_ = False)
            if len_ > 0:
                out_sec.append(l['sec'])
                out_len.append(len_)
                I.np.testing.assert_almost_equal(l['x_dist_end'] - (l['x_dist_start'] + l['x_offset']),
                                                 sec.L)
                # taking care of sections that started proximal of min_ but reach into slice
                x_offset = min_ - l['x_dist_start'] + l['x_offset']
                x_offset = max(0, x_offset)
                out_x_offset.append(x_offset)
        return out_sec, out_len, out_x_offset    

    def _map_single_synapse_in_interval(self, min_, max_, select = ['Dendrite', 'ApicalDendrite'], normed = False):
        d = self.d
        out_sec, out_len, out_x_offset  = self._get_sections_and_lengths_in_interval(min_, max_, select = select)
        id_, x = self._pick_random_point_on_segment(out_len)
        x = x + out_x_offset[id_]
        sec = out_sec[id_]
        sec_id = self.d.cell.sections.index(sec)
        if normed:
            x = x / sec.L
        return sec_id, x
    
    def _pick_random_point_on_segment(self, segment_lengths):
        segment_lengths_cumsum = I.np.cumsum(segment_lengths)
        r = I.np.random.rand() * segment_lengths_cumsum[-1]
        id_ = len(segment_lengths_cumsum[segment_lengths_cumsum < r])
        if id_ == 0:
            offset = 0
        else:
            offset = segment_lengths_cumsum[id_ - 1]
        return id_, r-offset
    
    def _get_n_syns_by_profile(self, density, n):
        return [self._pick_random_point_on_segment(density)[0] for lv in range(n)]
    
    def map_synapses_by_somadistance_profile(self, name, n, select = ['Dendrite', 'ApicalDendrite']):
        bins, density = self.profiles[name]
        out = []
        n_syn_per_bin = self._get_n_syns_by_profile(density, n)
        for bin_id in n_syn_per_bin:
            # print n_syn_per_bin
            syn = self._map_single_synapse_in_interval(bins[bin_id], bins[bin_id + 1], select = select)
            out.append(syn)
        return out
    
    def add_profile(self, name, bins, density, mode = 'soma_dist_micron', select = ['Dendrite', 'ApicalDendrite']):
        if mode == 'soma_dist_micron':
            pass
        elif mode == 'main_bifur':
            bifur = self.d.hot_zone_dist
            maxd = self.d._get_max_somadistance()
            # print bifur, maxd
            bins = unnormalize_bins(bins, bifur, maxd)
        else:            
            raise ValueError('mode must be soma_dist_micron or main_bifur')
        
        density_unnormalized = [] # density was normalized by amount of dendritic structure
        for lv in range(len(density)):
            factor = self.d.get_amount_of_dendrite_in_bin(bins[lv], bins[lv+1])
            density_unnormalized.append(float(density[lv]) * factor)
            
        self.profiles[name] = (bins,density_unnormalized)      
        
    
    
######################################
# class for loading cells
#####################################

class CellLoader():
    def __init__(self):
        # see here for preprocessing of the cells: 20190909_VPM_contact_analysis_for_mike_review.ipynb
        self.identifiers = [
                'WR54_Cell2_L5ST',
                'WR58_Cell5_L5TT_Final',
                'WR64_Cell8_L5TT_MikesASC',
                'WR69_Cell2_L5TT',
                'WR70_Cell3_L5TT_S3_Added_Moved_to_A1',
                'WR70_Cell5_L5TT',
                'WR71_ Cell6_L5TT',
                'WR77_lhs_Cell5_L5TT_TuftFixed',
                'WR81_Cell1_L5TT']
        
        self.suffixes =  {'unregistered': '.hoc', 'D2': '_D2-registered.hoc', 
            'landmark': '_Combination.landmarkAscii',
            'landmark_aligned': '_Combination_aligned.landmarkAscii',
            'landmark_D2': '_Combination_aligned_D2.landmarkAscii',
            'landmark_D2_top_bottom': '_Combination_aligned_D2_added_top_and_bottom_point.landmarkAscii'}
        
    def get_cell_object(self, id_, suffix):
        hocpath = mdb_cells['Data'].join(id_ + self.suffixes[suffix])
        return get_cell_object_from_hoc(hocpath)
    
    def get_synapses(self, id_):
        return mdb_cells[(id_, 'registered_points')]
 