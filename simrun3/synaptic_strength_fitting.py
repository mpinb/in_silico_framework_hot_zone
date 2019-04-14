'''This module has three parts.

First: class to manage synaptic strength fitting
Second: functions to simulate PSPs
Third: functions to analyze PSPs'''

import warnings
import Interface as I
activate_functional_synapse = I.scp.network.activate_functional_synapse
from .utils import get_cellnumbers_from_confile, split_network_param_in_one_elem_dicts
from .get_cell_with_network import get_cell_with_network

###############################
# First part: class to manage synaptic strength fitting
###############################
class PSPs:
    '''main class for calculation of ePSP amplitudes and synaptic strength fitting'''
    def __init__(self, neuron_param = None, confile = None, gExRange = [0.5, 1.0, 1.5, 2.0], 
                 AMPA_component = 1, NMDA_component = 1, vardt = True, save_vmax_dir = None):
        ''' 
        neuron_param: I.scp.NTParameterSet structure specifying the biophysical model
        confile: path to .con-file of the network realization that should be used
        gExRange: synaptic strength values to be checked
        vardt: should the variable step size solver be used=
        '''
        assert('neuron' in neuron_param.keys())
        if not save_vmax_dir is None:
            raise NotImplementedError()
        self.neuron_param = neuron_param
        self.confile = confile
        self.gExRange = gExRange
        self.vardt = vardt
        self.save_vmax_dir = save_vmax_dir
        self.AMPA_component = AMPA_component
        self.NMDA_component = NMDA_component
        self._keys = []
        self._delayeds = []
        self._simfun = I.dask.delayed(run_ex_synapses)
        self.futures = None
        self.result = None
        
        self.network_param = generate_ex_network_param_from_network_embedding(self.confile)
        self.network_params_by_celltype = split_network_param_in_one_elem_dicts(self.network_param)
        self._setup_computation()
        
    def _setup_computation(self):
        for n in self.network_params_by_celltype:
            assert(len(n.network.keys()) == 1)
            celltype = n.network.keys()[0]
            n.network[celltype]['synapses']
            for gEx in self.gExRange:
                self._keys.append((celltype, gEx*self.AMPA_component, gEx*self.NMDA_component))
                d = self._simfun(I.cloudpickle.dumps(self.neuron_param), I.cloudpickle.dumps(n), 
                                 celltype, gEx*self.AMPA_component, gEx*self.NMDA_component, vardt = self.vardt, 
                                 save_vmax_dir = self.save_vmax_dir)
                self._delayeds.append(d)
        
    def run(self, client, rerun = False):
        '''start running computation on dask.distributed.Client
        client: dask.distributedClient object
        
        rerun: If True, a computation can be restarted. Otherwise, 
                calling this function has no effect if the 
                computation is already running'''
        if (self.futures is None) or rerun:
            self.result = None
            self.futures = client.compute(self._delayeds)
    
    def get_voltage_traces(self):
        '''Returns voltage and timing of all synapse activations.
        Fetches results from the cluster.'''
        if self.result is None:
            self.result = [f.result() for f in self.futures]
            assert(len(self._keys) == len(self.result))   
            del self.futures
            self.futures = 'done'  
        out = I.defaultdict_defaultdict()
        for lv, k in enumerate(self._keys):
            out[k[0]][k[1]][k[2]] = self.result[lv]
            # calculate maximum voltage in the respective simulation
            # list comprehension used to flatten the list
            max = I.np.max([x for x in self.result[lv][3] for x in x])
            if  max > -45:
                errstr = "Result Nr {} has a maximum membrane potential of {} mV. ".format(lv, max) +\
                         "Make sure, the cell does not depolarize during initialization "+\
                         "as a suprathreashold activity can potentially change the PSP."
                print errstr
                if max > 0:
                    raise RuntimeError(errstr)
        return out
    
    def get_voltage_and_timing(self, method = 'dynamic_baseline'):
        '''calculate maxmimum and timing of the EPSPS
        method: dynamic_baseline: a simulation without any synaptic activation is 
            substracted from a simulation with cell activation. The maximum and 
            timepoint of maximum is returned
        method: constant_baseline: the voltage at t = 110ms (= directly before
            synapse activation) is considered as baseline and substracted
            from all voltages at all timepoints.
            The maximum and timepoint of the maximum after t = 110ms is 
            returned. 
        '''
        res = self.get_voltage_traces()
        if method == 'dynamic_baseline':
            vt = {k: {k: {k: [get_tMax_vMax_baseline(v[0], v[1], v[2][lv], v[3][lv])
                              for lv in range(len(v[2]))]
                          for k, v in v.iteritems()}
                      for k, v in v.iteritems()}
                  for k, v in res.iteritems()}
        elif method == 'constant_baseline':
            vt = {k: {k: {k: [get_tMax_vMax(v[2][lv], v[3][lv])
                              for lv in range(len(v[2]))]
                          for k, v in v.iteritems()}
                      for k, v in v.iteritems()}
                  for k, v in res.iteritems()}
        else:
            errstr = 'method must be dynamic_baseline or constant_baseline'
            raise ValueError(errstr)
        return vt
    
    def get_summary_statistics(self, method = 'dynamic_baseline',
                               merge_celltype_kwargs = {},
                               ePSP_summary_statistics_kwargs = {}):
        vt = self.get_voltage_and_timing(method)
        vt = merge_celltypes(vt, **merge_celltype_kwargs)
        return ePSP_summary_statistics(vt, **ePSP_summary_statistics_kwargs)
    
    def get_optimal_g(self, 
                      measured_data, 
                      method = 'dynamic_baseline'):
        pdf = self.get_summary_statistics(method = method)
        pdf = pdf.reset_index()
        pdf = pdf.groupby('celltype').apply(linear_fit_pdf)
        pdf = I.pd.concat([pdf, measured_data], axis = 1)
        calculate_optimal_g(pdf)
        return pdf
    
#############################################
# Second part: functions to simulate PSPs 
#############################################
def run_ex_synapse(cell_nw_generator, neuron_param, network_param, celltype, preSynCellID, gAMPA, gNMDA, vardt = False):
    '''core function, that actually activates a single synapse and runs the simulation.
    cell_nw_generator: simrun3.get_cell_with_network.get_cell_with_network
    neuron_param: single_cell_parser.NTParameterSet specifying biophysical properties
    network_param: single_cell_parser.NTParameterSet specifying network properties
    celltype: presynaptic celltype
    preSynCellID: number of presynaptic cell that should be activated. None: No cell will be activated.
    gEx: conductance of AMPA and NMDA component
    vardt: whether variable stepsize solver should be used '''
    tOffset = 100
    spikeTime = 10 
    
    cell, nwMap = cell_nw_generator()
    
    for cellType in nwMap.cells.keys():
        for syn in cell.synapses[cellType]:
            syn.disconnect_hoc_synapse()
    
    synParameters = network_param.network[celltype]['synapses']
    
    for syn in cell.synapses[celltype]:             
        syn.weight = {'glutamate_syn': [gAMPA, gNMDA]}     
    
    if preSynCellID is not None:
        preSynCell = nwMap.cells[celltype][preSynCellID]
        for syn in preSynCell.synapseList:
            syn.weight = {'glutamate_syn': [gAMPA, gNMDA]}  
            activate_functional_synapse(syn, cell, preSynCell, synParameters, releaseTimes = [tOffset+spikeTime])
    
    neuron_param.sim.tStop = 150
    
    I.scp.init_neuron_run(neuron_param.sim, vardt = vardt)
    t,v = I.np.array(cell.tVec), I.np.array(cell.soma.recVList)[0,:]
    
    # without the following lines, the simulation will crash from time to time
    cell.re_init_cell()
    nwMap.re_init_network()
    
    return t,v

def run_ex_synapses(neuron_param, network_param, celltype, gAMPDA, gNMDA, vardt = False, 
                    save_vmax_dir = None):
    '''method to consecutively calculate all EPSPs of all presynaptic cells of one celltype.'''
    if isinstance(neuron_param, str):
        neuron_param = I.cloudpickle.loads(neuron_param)
    if isinstance(network_param, str):
        network_param = I.cloudpickle.loads(network_param)    
    with I.silence_stdout:
        cell_nw_generator = get_cell_with_network(neuron_param, network_param)
        cell, nwMap = cell_nw_generator()
    somaT, somaV, = [], []
    t_baseline, v_baseline = run_ex_synapse(cell_nw_generator, neuron_param, network_param, 
                                            celltype, None, gAMPDA, gNMDA, vardt = vardt)
    for preSynCellID in range(len(nwMap.connected_cells[celltype])):
        t,v = run_ex_synapse(cell_nw_generator, neuron_param, network_param, 
                             celltype, preSynCellID, gAMPDA, gNMDA, vardt = vardt)
        somaT.append(t), somaV.append(v)
    print 'deleting cell_nw_generator, cell, nwMap'
    del cell_nw_generator, cell, nwMap
    return t_baseline, v_baseline, somaT, somaV

def generate_ex_network_param_from_network_embedding(confile):
    '''returns a network parameter file that can be used to activate all presynaptic 
    conected cells as specified in confile:'''
    param_template = {'glutamate_syn': {'delay': 0.0,
      'parameter': {'decaynmda': 1.0,
                    'facilampa': 0.0,
                    'facilnmda': 0.0,
                    'tau1': 26.0,
                    'tau2': 2.0,
                    'tau3': 2.0,
                    'tau4': 0.1},
      'threshold': 0.0,
      'weight': [0.0, 1.0]}}
    
    out = I.defaultdict_defaultdict()
    for k, cellnumber in get_cellnumbers_from_confile(confile).iteritems():
        if not k.split('_')[0] in I.excitatory:
            continue
        out['network'][k]['cellNr'] = cellnumber
        out['network'][k]['activeFrac'] = 1.0
        out['network'][k]['celltype'] = 'pointcell'
        out['network'][k]['spikeNr'] = 1
        out['network'][k]['spikeT'] = 10
        out['network'][k]['spikeWidth'] = 1.0
        out['network'][k]['synapses']['connectionFile'] = confile
        out['network'][k]['synapses']['distributionFile'] = confile[:-3] + 'syn'
        out['network'][k]['synapses']['receptors'] = param_template
    return I.scp.NTParameterSet(out) 

###############################################
# Third part: functions to analyze PSPs
###############################################

def get_tMax_vMax_baseline(t_baseline,v_baseline, t,v):
    '''this method calculates the ePSP amplitude by subtracting a voltage trace without
    any synapse activation from a voltage trace with synapse activation.
    t_basline, v_baseline: voltage trace without synapse activation
    t,v: voltage trace with synapse activation'''
    try:
        t,v = I.np.arange(0,150,0.025), I.np.interp(I.np.arange(0,150,0.025), t, v)
    except:
        raise RuntimeError() 
    t_baseline,v_baseline = I.np.arange(0,150,0.025), I.np.interp(I.np.arange(0,150,0.025), t_baseline, v_baseline)
    return I.sca.analyze_voltage_trace(v-v_baseline, t)

def get_tMax_vMax(t,v):
    '''This method calculates the ePSP amplitude by subtracting the voltage
    directly before the epsp, which is supposed to be at 160ms, from the voltagetrace.'''
    t,v = I.np.arange(0,150,0.025), I.np.interp(I.np.arange(0,150,0.025), t, v) 
    baseline = I.np.median(v[4360:4400]) # 1ms pre-stim
    v -= baseline
    return I.sca.analyze_voltage_trace(v[4400:], t[4400:])

def merge_celltypes(vt, 
                    detection_strings = ['L2', 'L34', 'L4', 'L5st', 'L5tt', 'L6cc', 'L6ct', 'VPM_C2'], 
                    celltype_must_be_in = I.excitatory):
    '''concatenates voltagetraces of celltypes'''
    out = I.defaultdict_defaultdict()
    for detection_string in detection_strings:
        for celltype in vt.keys():
            if not celltype.split('_')[0] in celltype_must_be_in:
                print 'skipping {}'.format(celltype)
                continue
            for gAMPA in vt[celltype].keys():
                for gNMDA in vt[celltype][gAMPA].keys():
                    if detection_string in celltype:
                        if not isinstance(out[detection_string][gAMPA][gNMDA], list):
                            out[detection_string][gAMPA][gNMDA] = []
                        out[detection_string][gAMPA][gNMDA].extend(vt[celltype][gAMPA][gNMDA])
    return out

def ePSP_summary_statistics(vt, threashold = 0.1, tPSPStart = 100.0):
    summaryData = []#I.defaultdict_defaultdict()
    for celltype, vt in vt.iteritems():
        for gAMPA, vt in vt.iteritems():
            for gNMDA, vt in vt.iteritems():
                t = list(v[0] for v in vt if v[1] >= threashold)                
                v = list(v[1] for v in vt if v[1] >= threashold)
                if len(t) == 0:
                    print("skipping celltype {}, gAMPA {}, gNMDA {}: no response above threashold of {} found".format(celltype, gAMPA, gNMDA, threashold))
                    continue
                out = {}                    
                out['epspMean'] = I.np.mean(v)
                out['epspStd'] = I.np.std(v)
                out['epspMed'] = I.np.median(v)
                out['epspMin'] = I.np.min(v)
                out['epspMax'] = I.np.max(v)
                out['tMean'] = I.np.mean(I.np.array(t)-tPSPStart)
                out['tStd'] = I.np.std(I.np.array(t)-tPSPStart)
                out['tMed'] = I.np.median(I.np.array(t)-tPSPStart)
                out['gNMDA'] = gNMDA
                out['gAMPA'] = gAMPA
                out['celltype'] = celltype
                summaryData.append(out)
    return I.pd.DataFrame(summaryData).set_index(['celltype', 'gAMPA', 'gNMDA'])

def linear_fit(gAMPANMDA, epsp):
    return I.np.polyfit(gAMPANMDA, epsp, 1)

def linear_fit_pdf(pdf):
    assert(pdf['gAMPA'].equals(pdf['gNMDA']))
    return I.pd.Series({'EPSP mean_offset': linear_fit(pdf.gAMPA, pdf.epspMean)[1],
                                       'EPSP mean_slope': linear_fit(pdf.gAMPA, pdf.epspMean)[0],
                                       'EPSP std_offset': linear_fit(pdf.gAMPA, pdf.epspStd)[1],
                                       'EPSP std_slope': linear_fit(pdf.gAMPA, pdf.epspStd)[0],
                                       'EPSP med_offset': linear_fit(pdf.gAMPA, pdf.epspMed)[1],
                                       'EPSP med_slope': linear_fit(pdf.gAMPA, pdf.epspMed)[0],
                                       'EPSP max_offset': linear_fit(pdf.gAMPA, pdf.epspMax)[1],
                                       'EPSP max_slope': linear_fit(pdf.gAMPA, pdf.epspMax)[0]})

def calculate_optimal_g(pdf):
    mean = (pdf['EPSP_mean_measured']-pdf['EPSP mean_offset']) / pdf['EPSP mean_slope']
    med = (pdf['EPSP_med_measured']-pdf['EPSP med_offset']) / pdf['EPSP med_slope']
    max_ = (pdf['EPSP_max_measured']-pdf['EPSP max_offset']) / pdf['EPSP max_slope']
    pdf['optimal g'] = (2 * mean + 2 * med + 1 * max_ ) * 1 / 5.
    return 