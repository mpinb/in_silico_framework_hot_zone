import Interface as I
import warnings
def get_binsize(length, binsize_goal = 50):
    '''given a length of a branch, returns number of binsize and number of bins that results in binning closest to 
    the binsize goal'''
    n_bins = length / float(binsize_goal)
    n_bins_lower = I.np.floor(n_bins)
    n_bins_lower = I.np.max([n_bins_lower, 1])
    n_bins_upper = I.np.ceil(n_bins)
    binsize_lower = length / n_bins_lower
    binsize_upper = length / n_bins_upper
    if I.np.abs(binsize_goal - binsize_lower)  < I.np.abs(binsize_goal - binsize_upper):
        return binsize_lower, int(n_bins_lower)
    else:
        return binsize_upper, int(n_bins_upper)
    
def get_bin(value, bin_min, bin_max, n_bins, tolerance = 0.1):
    bin_size = (bin_max - bin_min) / n_bins
    bins = [bin_min + n * bin_size for n in range(n_bins+1)]
    bins[0] = bins[0]-tolerance
    bins[-1] = bins[-1]+tolerance
    if (I.np.max(value) >= bins[-1]) or (I.np.min(value) <= bins[0]) :
        raise RuntimeError('value outside of bin_min and bin_max!')
    return I.np.digitize(value, bins)

def get_neuron_param_file(m):
    folder = m['parameterfiles_cell_folder']
    f = [f for f in folder.listdir() if not f.endswith('.pickle')]#+.keys()
    if not len(f) == 1:
        warnings.warn('found more than one neuron parameter file in the database. Make sure the database does not contain simulations with more than one morphology!')
    return folder.join(f[0])

def get_section_distances_df(neuron_param_file, silent = True):
    neup = I.scp.NTParameterSet(neuron_param_file)
    if silent:
        with I.silence_stdout:
            cell = I.scp.create_cell(neup.neuron)    
    sections_min_dist = [I.sca.synanalysis.compute_distance_to_soma(sec, 0) for sec in cell.sections]
    sections_max_dist = [I.sca.synanalysis.compute_distance_to_soma(sec, 1) for sec in cell.sections]
    binsize = [get_binsize(s_ma-s_mi)[0] if (cell.sections[lv].label != 'Soma') else 'Soma' for lv, (s_mi, s_ma) in enumerate(zip(sections_min_dist, sections_max_dist))]
    n_bins = [get_binsize(s_ma-s_mi)[1] if (cell.sections[lv].label != 'Soma') else 'Soma' for lv, (s_mi, s_ma) in enumerate(zip(sections_min_dist, sections_max_dist))]
    bin_borders = [I.np.linspace(s_mi, s_ma, num = n_bins) if not isinstance(n_bins, str) else 'Soma' for s_mi, s_ma, n_bins in 
                  zip(sections_min_dist, sections_max_dist, n_bins)]
    section_distances_df = I.pd.DataFrame({'min_': sections_min_dist, 'max_': sections_max_dist, 'n_bins': n_bins, 'binsize': binsize})
    return section_distances_df

def get_spatial_bin_names(section_distances_df):
    """Given a dataframe describing the distance to soma of all sections, as provided by :@function get_section_distances_df:,
    this method returns the names for all the spatial bins of the format section_id/bin_id. E.g.: 5/2 denotes the second bin of section 5.


    Args:
        section_distances_df (pd.DataFrame): the dataframe describing distance to soma for all sections, as provided by :@function get_section_distances_df:

    Returns:
        list: A list containing the bin ids in string format.
    """
    all_bins = []
    for index, row in section_distances_df.iterrows():
        n_bins = row['n_bins']
        if n_bins == 'Soma':
            all_bins.append(str(index) + '/' + str(0))
        else:
            for n in range(n_bins):
                all_bins.append(str(index) + '/' + str(n+1))
    return all_bins

def get_bin_soma_distances_in_section(section_id, section_distances_df):
    """Given a section id, this method returns the distance to soma for all bins in this section

    Args:
        section_id (int): Id of the neuron section
        section_distances_df (pd.DataFrame): the dataframe describing distance to soma for all sections, as provided by :@function get_section_distances_df:

    Returns:
        list: A list containing the distance to soma for each bin in a section.
    """
    section = section_distances_df.iloc[int(section_id)]
    soma_d = []
    for i in range(section["n_bins"]):
        # centers of the bins
        soma_d.append(section["min_"] + int(i+0.5)*(section["max_"] - section["min_"])/section["n_bins"])
    return soma_d

def get_bin_adjacency_map_in_section(cell, section_id, section_distances_df):
    """Fetches all the sections neighboring the given section id.
    It then checks which bins are adjacent within these sections.
    Sections are defined by the neuron simulation, but bins are ad-hoc defined by data preparation for the ANN.
    Eeach section always has one or more bins.

    
    Args:
        section_id (int): index of the neuron section
        section_distances_df (pd.DataFrame): the dataframe describing distance to soma for all sections, as provided by :@function get_section_distances_df:


    Returns:
        dict: a dictionary with pairs of adjacant bins.
    """
    neighboring_bins = {}
    neighbor_map = cell.get_section_adjacancy_map()
    for parent in neighbor_map[int(section_id)]["parents"]:
        # section_id/1 is connected to some parent section, but which bin?
        distances = get_bin_soma_distances_in_section(parent, section_distances_df)
        d_of_this_bin = get_bin_soma_distances_in_section(section_id, section_distances_df)[0]
        diff = [abs(parent_d - d_of_this_bin) for parent_d in distances]
        closest = I.np.argmin(diff)
        neighboring_bins["{}/{}".format(section_id, 1)] = "{}/{}".format(parent, closest+1)
    for child in neighbor_map[int(section_id)]["children"]:
        # children/1 is connected to some bin of the current section, but which bin?
        distances = get_bin_soma_distances_in_section(section_id, section_distances_df)
        d_of_child_bin = get_bin_soma_distances_in_section(child, section_distances_df)[0]
        diff = [abs(d - d_of_child_bin) for d in distances]
        closest = I.np.argmin(diff)
        neighboring_bins["{}/{}".format(child, 1)] = "{}/{}".format(section_id, closest+1)
    # assure mutuality
    neighboring_bins_copy = neighboring_bins.copy()
    for key, val in neighboring_bins_copy.items():
        neighboring_bins[val] = key
    return neighboring_bins

def get_neighboring_spatial_bins(cell, section_distances_df, bin_id):
    """Given a bin id, this method returns all the neighboring bins

    Args:
        cell (cell): cell object
        neup (_type_): neuron parameter file
        bin_id (_type_): BIn id of format: section_id/bin_id

    Returns:
        list: list of all ids of bins that neighbor the given bin
    """
    neighbors = []
    section_id_, bin_id_ = map(int, bin_id.split('/'))
    n_bins = section_distances_df.iloc[int(section_id_)]['n_bins']
    assert 0 < int(bin_id_) <= n_bins, "Bin does not exist. Section {} only has {} bins, but you asked for bin #{}".format(section_id_, n_bins, bin_id_)
    if 1 < int(bin_id_):
        # append previous bin
        neighbors.append('{}/{}'.format(section_id_, int(bin_id_) - 1))
    if int(bin_id_) < n_bins:
        # append next bin
        neighbors.append('{}/{}'.format(section_id_, int(bin_id_) + 1))
    # check for adjacent sections
    bin_adj = get_bin_adjacency_map_in_section(cell, section_id_, section_distances_df)
    if bin_id in bin_adj:
        neighbors.append(bin_adj[bin_id])
    return neighbors

def augment_synapse_activation_df_with_branch_bin(sa_, 
                                                  section_distances_df = None, 
                                                  synaptic_weight_dict = None):
    out = []
    for secID, df in sa_.groupby('section_ID'):
        min_ = section_distances_df.loc[secID]['min_']
        max_ = section_distances_df.loc[secID]['max_']
        n_bins = section_distances_df.loc[secID]['n_bins']
        df = df.copy()    
        if n_bins == 'Soma':
            df['branch_bin'] = 0
        else:
            df['branch_bin'] = get_bin(df.soma_distance, min_, max_, n_bins)
        out.append(df)
    sa_faster = I.pd.concat(out).sort_values(['synapse_type', 'synapse_ID']).sort_index()
    sa_faster['section/branch_bin'] = sa_faster['section_ID'].astype(str) + '/' + sa_faster['branch_bin'].astype('str')
    sa_faster['celltype'] = sa_faster.synapse_type.str.split('_').str[0]
    sa_faster['EI'] = sa_faster['celltype'].isin(I.excitatory).replace(True, 'EXC').replace(False, 'INH')
    if synaptic_weight_dict:
        sa_faster['syn_weight'] = sa_faster['synapse_type'].map(syn_weights)
    return sa_faster

@I.dask.delayed
def save_VT_batch(vt_batch, batch_id, outdir = None, fname_template = 'batch_{}_VT_ALL.npy'):
    fname = fname_template.format(batch_id) 
    outpath = I.os.path.join(outdir, fname)
    I.np.save(outpath, vt_batch)
    
@I.dask.delayed
def save_st_and_ISI(st, batch_id, chunk, min_time, max_time, outdir,suffix = 'SOMA'):
    current_st = st.loc[chunk]
    ISI_array = compute_ISI_array(current_st, min_time, max_time)
    AP_array = compute_AP_array(current_st,min_time, max_time)
    I.np.save(outdir.join('batch_{}_AP_{}.npy').format(batch_id,suffix), AP_array)
    I.np.save(outdir.join('batch_{}_ISI_{}.npy').format(batch_id,suffix), ISI_array)
    
def compute_ISI_from_st(st, timepoint, fillna = None):
    '''suggestion: use the temporal window width as fillna'''
    assert(fillna is not None)
    st = st.copy()
    st[st>timepoint] = I.np.NaN                       # set all spike times beyond timepoint to NaN
    max_spike_time_before_timepoint = st.max(axis=1)
    ISI = timepoint - max_spike_time_before_timepoint
    if fillna is not None:
        ISI = ISI.fillna(fillna)
    return ISI

def compute_ISI_array(st, min_time, max_time, fillna = 1000):
    ISI = []
    for t in range(min_time, max_time):
        ISI.append(compute_ISI_from_st(st, t, fillna).to_numpy().reshape(-1,1))
    return I.np.concatenate(ISI, axis = 1)

def compute_AP_array(st, min_time, max_time, fillna = 1000):
    AP = []
    for i in range(min_time, max_time):
        AP.append(I.spike_in_interval(st, i,i +1).to_numpy().reshape(-1,1))
    return I.np.concatenate(AP, axis = 1)

def load_syn_weights(m, client):
    folder_ = m['parameterfiles_network_folder']
    fnames = m['parameterfiles_network_folder'].listdir()
    fnames.pop(fnames.index('Loader.pickle'))
    param_df = m['parameterfiles']
    syn_weights_out = {}
    
    @I.dask.delayed
    def _helper(fname, stis):
        stis = I.cloudpickle.loads(stis)
        syn_weights = {}
        netp = I.scp.build_parameters(folder_.join(fname))
        for celltype in netp.network:
            if celltype.split('_')[0] in I.excitatory:
                syn_weights[celltype] = netp.network[celltype].synapses.receptors.glutamate_syn.weight[0]
            else:
                syn_weights[celltype] = netp.network[celltype].synapses.receptors.gaba_syn.weight
        return {sti: syn_weights for sti in stis}
    
    stis_by_netp = {k:list(v.index) for k,v in param_df.groupby('hash_network')}
    delayeds = []
    for f in fnames: 
        d = _helper(f, I.cloudpickle.dumps(stis_by_netp[f]))
        delayeds.append(d)

    result = client.gather(client.compute(delayeds))
    
    syn_weights_out = {}
    for r in result:
        syn_weights_out.update(r)
    
    return syn_weights_out

def temporal_binning_augmented_sa(sa_augmented, min_time, max_time, bin_size, use_weights = False):
    activation_times = sa_augmented[[c for c in sa_augmented.columns if I.utils.convertible_to_int(c)]].values
    if use_weights:
        weights = (activation_times*0+1)*sa_augmented.syn_weight.values.reshape(-1,1)
        weights = weights[~I.np.isnan(activation_times)]
    activation_times = activation_times[~I.np.isnan(activation_times)]
    bin_borders = I.np.arange(min_time, max_time + bin_size, bin_size)
    if use_weights:
        return bin_borders, I.np.histogram(activation_times, bin_borders, weights = weights)[0]  
    else:
        return bin_borders, I.np.histogram(activation_times, bin_borders)[0]

def get_synapse_activation_array_weighted(sa_, selected_stis = None, spatial_bin_names = None,
                                 min_time = 0, max_time = 600, bin_size = 1, 
                                 use_weights = False):
    fun = I.partial(temporal_binning_augmented_sa, 
                    min_time = min_time, 
                    max_time = max_time, 
                    bin_size = 1, 
                    use_weights = use_weights)
    # sa_augmented = augment_synapse_activation_df_with_branch_bin(sa_, section_distances_df = section_distances_df)
    synapse_activation_binned = sa_.groupby([sa_.index, 'EI','section/branch_bin']).apply(lambda x: fun(x)[1])
    synapse_activation_binned_dict = synapse_activation_binned.to_dict()
    n_time_bins = len(list(synapse_activation_binned_dict.values())[0])
    level_0_values = selected_stis
    level_1_values = ['EXC', 'INH']
    level_2_values = spatial_bin_names
    array = I.np.zeros((len(level_0_values), len(level_1_values), len(level_2_values), n_time_bins))
    for i0, l0 in enumerate(level_0_values):
        for i1, l1 in enumerate(level_1_values):
            for i2, l2 in enumerate(level_2_values):
                if not (l0,l1,l2) in synapse_activation_binned_dict.keys():
                    continue
                arr = synapse_activation_binned_dict[(l0,l1,l2)]
                array[i0,i1,i2,:]=arr
    return array

@I.dask.delayed
def save_SA_batch(sa_, 
                         selected_stis, 
                         batch_id,
                         outdir = None,
                         section_distances_df = None,
                         spatial_bin_names = None,
                         min_time = 0, 
                         max_time = 600, 
                         bin_size = 1,
                         synaptic_weight_dict = None):
    syn_weights = None
    if syn_weights:
        raise NotImplementedError()
        fname = 'batch_{}_SYNAPSE_ACTIVATION_WEIGHTED.npy'.format(batch_id) 
    else: 
        fname = 'batch_{}_SYNAPSE_ACTIVATION.npy'.format(batch_id) 
    outpath = I.os.path.join(outdir, fname)
    sa_ = augment_synapse_activation_df_with_branch_bin(sa_, section_distances_df, syn_weights)
    array = get_synapse_activation_array_weighted(sa_, selected_stis, spatial_bin_names = spatial_bin_names,
                                            min_time = min_time, max_time = max_time, bin_size = bin_size,
                                            use_weights = syn_weights is not None)        
    I.np.save(outpath, array)

# code to get the maximum depolarization per ms of a voltage traces dataframe  
def get_time_groups(vt_pandas):
        time_groups = []
        for lv in range(len(vt_pandas.columns)):
            time_groups.append(lv//40)
        return time_groups

def get_max_per_ms_on_pandas_dataframe(vt_pandas):
    time_groups = get_time_groups(vt_pandas)
    vt_max = vt_pandas.groupby(time_groups, axis = 1).apply(lambda x: x.max(axis = 1))
    return vt_max

def get_max_depolarization_per_ms(vt_dask):
    meta_ = {t:'float64' for t in get_time_groups(vt_dask.head())}
    vt_max_dask = vt_dask.map_partitions(get_max_per_ms_on_pandas_dataframe, meta = meta_)
    return vt_max_dask
        
# initializer class
class Init:
    def __init__(self, mdb,
         mdb_target = None,
         min_time = None, 
         max_time = None, 
         bin_size = 1,
         batchsize = 500, 
         client = None,
         sti_selection = None, 
         sti_selection_name = None,
         persist = False,
         synaptic_weight = False,
         dend_ap_suffix = '_-30.0'):
        
        if sti_selection is not None:
            assert(sti_selection_name is not None)
        if mdb_target is None:
            mdb_target = mdb
        
        self.mdb = mdb
        self.mdb_target = mdb_target
        self.min_time = min_time
        self.max_time = max_time
        self.bin_size = bin_size
        self.batchsize = batchsize
        self.client = client
        self.sti_selection = sti_selection
        self.sti_selection_name = sti_selection_name
        self.persist = persist
        self.dend_ap_suffix = dend_ap_suffix
        
        # get selected trials - if none specified, select all
        if sti_selection is not None:
            sti_selection = list(sti_selection)
        else:
            st = mdb['spike_times']
            sti_selection = list(st.index)

        self.chunks = I.utils.chunkIt(sti_selection, len(sti_selection)/batchsize)
        
        # create directory to save batches
        if sti_selection_name:
            x = '{}_{}_{}_'.format(min_time, max_time, bin_size) + sti_selection_name
        else:
            x = '{}_{}_{}_'.format(min_time, max_time, bin_size) + 'all'
        self.outdir = mdb_target.create_managed_folder(('ANN_batches', x, 'EI__section/branch_bin'), raise_ = False)
        
        # setup spatial bins
        self._setup_spatial_bins()
        
        self._setup_synaptic_weights()
    
    def _setup_spatial_bins(self):
        self.neuron_param_file = get_neuron_param_file(self.mdb)
        self.section_distances_df = get_section_distances_df(self.neuron_param_file)
        self.spatial_bin_names = get_spatial_bin_names(self.section_distances_df)
        
    def _setup_synaptic_weights(self):
        self.synaptic_weight_dict = None
        # if synaptic_weight:
        #     synaptic_weight_dict = load_syn_weights(m,self.client)
        #     synaptic_weight_dict = self.client.persist(synaptic_weight_dict)
        # else:
        #     synaptic_weight_dict = None
        
    def init_all(self,
                init_soma_vt = True,
                init_dend_vt = True,
                init_soma_AP_ISI = True,
                init_dend_AP_ISI = True,
                init_max_soma_vt = True,
                init_max_dend_vt = True,
                init_synapse_activation = True,
                init_synapse_activation_weighted = True
                ):
        pass
    
    def _get_distal_recording_site(self):
        keys = self.mdb['dendritic_recordings'].keys()
        dist_rec_site = sorted(keys, key = lambda x: float(x.split('_')[-1]))[1]
        return dist_rec_site
    
    def _save_vt(self, vt, fname_template = None):
        delayeds = []
        for batch_id, chunk in enumerate(self.chunks):
            selected_indices = chunk
            d = save_VT_batch(vt.loc[selected_indices],
                             batch_id,
                             outdir = self.outdir,
                             fname_template = fname_template)
            delayeds.append(d)
        return delayeds
            
    def init_soma_vt(self):
        vt = self.mdb['voltage_traces'].iloc[:,::40].iloc[:,self.min_time:self.max_time]
        return self._save_vt(vt, fname_template = 'batch_{}_VT_SOMA.npy')
        
    def init_dend_vt(self):        
        dist_rec_site = self._get_distal_recording_site()
        vt_dend = self.mdb['dendritic_recordings'][dist_rec_site]
        vt_dend = vt_dend.iloc[:,::40].iloc[:,self.min_time:self.max_time]
        return self._save_vt(vt_dend, fname_template = 'batch_{}_VT_DEND.npy')

    def _save_AP_ISI(self, st, suffix = None):
        delayeds = []
        # st = self.client.scatter(st)
        for batch_id, chunk in enumerate(self.chunks):
            d = save_st_and_ISI(st, batch_id, chunk, self.min_time, self.max_time, self.outdir, suffix = suffix)
            delayeds.append(d)
        return delayeds
    
    def init_soma_AP_ISI(self):
        st = self.client.scatter(self.mdb['spike_times'])
        return self._save_AP_ISI(st,suffix = 'SOMA')

    def init_dend_AP_ISI(self):
        dist_rec_site = self._get_distal_recording_site()
        st = self.client.scatter(self.mdb['dendritic_spike_times'][dist_rec_site + self.dend_ap_suffix])
        return self._save_AP_ISI(st, suffix = 'DEND')
    
    def init_max_soma_vt(self):
        vt = self.mdb['voltage_traces']
        vt = get_max_depolarization_per_ms(vt)
        vt = vt.iloc[:,self.min_time:self.max_time]
        return self._save_vt(vt, fname_template = 'batch_{}_VT_SOMA_MAX.npy')
    
    def init_max_dend_vt(self):
        dist_rec_site = self._get_distal_recording_site()
        vt_dend = self.mdb['dendritic_recordings'][dist_rec_site]
        vt_dend = get_max_depolarization_per_ms(vt_dend)
        vt_dend = vt_dend.iloc[:,self.min_time:self.max_time]
        return self._save_vt(vt_dend, fname_template = 'batch_{}_VT_DEND_MAX.npy')
    
    def init_synapse_activation(self, synaptic_weight = False):
        sa = self.mdb['synapse_activation']
        if self.persist: 
            sa = self.client.persist(sa)
            
        delayeds = []
        print('Create delayed objects for synapse_activations')

        for batch_id, chunk in enumerate(self.chunks):
            selected_indices = chunk
            d = save_SA_batch(sa.loc[chunk],
                             chunk, 
                             batch_id,
                             outdir = self.outdir,
                             section_distances_df = self.section_distances_df,
                             spatial_bin_names = self.spatial_bin_names,
                             min_time = self.min_time, 
                             max_time = self.max_time, 
                             bin_size = self.bin_size,
                             synaptic_weight_dict = self.synaptic_weight_dict)
            delayeds.append(d)
        return delayeds
        
    
    def init_synapse_activation_weighted(self):
        delayeds = []
        return delayeds
    
def init(mdb,
         mdb_target = None,
         min_time = None, 
         max_time = None, 
         bin_size = 1,
         batchsize = 500, 
         client = None,
         sti_selection = None, 
         sti_selection_name = None,
         persist = False,
         synaptic_weight = False,
         dend_ap_suffix = '_-30.0',
         ):
    '''persist: whether the synapse activation dataframe should be loaded into memory 
                on the dask cluster. Speeds up the process if it fits in memory, otherwise leads to crash'''
    if mdb_target is None:
        mdb_target = mdb
        
    if sti_selection is not None:
        assert(sti_selection_name is not None)
    
    sa = mdb['synapse_activation']
    st = mdb['spike_times']
    if sti_selection is not None:
        sti_selection = list(sti_selection)
    else:
        sti_selection = list(st.index)
        
    chunks = I.utils.chunkIt(sti_selection, len(sti_selection)/batchsize)
    
    neuron_param_file = get_neuron_param_file(mdb)
    section_distances_df = get_section_distances_df(neuron_param_file)
    spatial_bin_names = get_spatial_bin_names(section_distances_df)
    if sti_selection_name:
        x = '{}_{}_{}_'.format(min_time, max_time, bin_size) + sti_selection_name
    else:
        x = '{}_{}_{}_'.format(min_time, max_time, bin_size) + 'all'
    outdir = mdb_target.create_managed_folder(('ANN_batches', x, 'EI__section/branch_bin'), raise_ = False)
    
    if persist: 
        sa = client.persist(sa)
    
    if synaptic_weight:
        synaptic_weight_dict = load_syn_weights(m,client)
        synaptic_weight_dict = client.persist(synaptic_weight_dict)
    else:
        synaptic_weight_dict = None
        
    delayeds = []
    print('Create delayed objects for synapse_activations')

    for batch_id, chunk in enumerate(chunks):
        selected_indices = chunk
        d = save_SA_batch(sa.loc[chunk],
                         chunk, 
                         batch_id,
                         outdir = outdir,
                         section_distances_df = section_distances_df,
                         spatial_bin_names = spatial_bin_names,
                         min_time = min_time, 
                         max_time = max_time, 
                         bin_size = bin_size,
                         synaptic_weight_dict = synaptic_weight_dict)
        delayeds.append(d)
    
    print('Create delayed objects for somatic voltage traces')
    # somatic voltage traces
    vt = mdb['voltage_traces'].iloc[:,::40].iloc[:,min_time:max_time]
    for batch_id, chunk in enumerate(chunks):
        selected_indices = chunk
        d = save_VT_batch(vt.loc[selected_indices],
                         batch_id,
                         outdir = outdir)
        delayeds.append(d)

    print('Create delayed objects for somatic APs and ISIs')
    # somatic AP and ISI
    st = client.scatter(mdb['spike_times'])
    for batch_id, chunk in enumerate(chunks):
        d = save_st_and_ISI(st, batch_id, chunk, min_time, max_time, outdir)
        delayeds.append(d)
        
    # dendritic voltage traces
    print('Create delayed objects for dendritic voltage traces')
    keys = mdb['dendritic_recordings'].keys()
    dist_rec_site = sorted(keys, key = lambda x: float(x.split('_')[-1]))[1]
    
    vt_dend = mdb['dendritic_recordings'][dist_rec_site]
    vt_dend = vt_dend.iloc[:,::40].iloc[:,min_time:max_time]
    for batch_id, chunk in enumerate(chunks):
        selected_indices = chunk
        d = save_VT_batch(vt_dend.loc[selected_indices],
                         batch_id,
                         outdir = outdir,
                         fname_template = 'batch_{}_VT_DEND_ALL.npy')
        delayeds.append(d)
    
    # dend AP and ISI
    print('Create delayed objects for dendriticAPs and ISIs')
    st = client.scatter(mdb['dendritic_spike_times'][dist_rec_site + dend_ap_suffix])
    for batch_id, chunk in enumerate(chunks):
        d = save_st_and_ISI(st, batch_id, chunk, min_time, max_time, outdir,suffix = 'DEND')
        delayeds.append(d)
        
    return delayeds

def run_delayeds_incrementally(client, delayeds):
    import time
    futures = []
    ncores = sum(client.ncores().values())
    for ds in I.utils.chunkIt(delayeds, len(delayeds)/ncores):
        f = client.compute(ds)
        futures.extend(f)
        while True:
            time.sleep(1)
            futures = [f for f in futures if not f.status == 'finished']
            for f in futures:
                if f.status == 'error':
                    raise RuntimeError()
            if len(futures) < ncores*2:
                break
    I.distributed.wait(futures)
    return futures