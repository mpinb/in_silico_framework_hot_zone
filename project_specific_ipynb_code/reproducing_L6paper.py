'''THis module contains code to facilitate computations in the ipynb files that 
contain the name '_reproducing_L6paper_ in /nas1/Data_arco/notebooks'''

import Interface as I
from simrun2.reduced_model.get_kernel import compare_lists_by_none_values
from single_cell_parser.network_param_modify_functions import set_stim_onset
from single_cell_parser.network_param_modify_functions import change_ongoing_interval
from single_cell_parser import network
import biophysics_fitting.utils
from biophysics_fitting.hay_evaluation import objectives_BAC, objectives_step


###########################
# setting up model data base
###########################

class L6config:
    def __init__(self, 
                 mdb = None, 
                 biophysical_model_mdb = None, 
                 biophysical_model_mdb_key = None,
                 anatomical_model_mdb = None,
                 anatomical_model_mdb_key = None,
                 hocpath = None,
                 runs = None,
                 locs = ['C2center', 'C1border', 'C3border', 'B1border', 'B2border', 'B3border', 'D1border', 'D2border', 'D3border'],
                 stims = ['B1', 'B2', 'B3', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3', 'E2']):
        assert(mdb is not None)
        assert(biophysical_model_mdb is not None)
        assert(biophysical_model_mdb_key is not None)
        assert(anatomical_model_mdb is not None)        
        assert(runs is not None)
        assert(hocpath is not None)
        self.mdb = mdb
        self.biophysical_model_mdb = biophysical_model_mdb
        self.biophysical_model_mdb_key = biophysical_model_mdb_key
        self.anatomical_model_mdb = anatomical_model_mdb 
        if anatomical_model_mdb_key is not None:
            self.anatomical_model_mdb_key = anatomical_model_mdb_key    
        else:
            self.anatomical_model_mdb_key = biophysical_model_mdb_key    
        self.runs = runs
        self.hocpath = hocpath
        self.locs = locs
        self.stims = stims
        
    def setup_mdb(self):
        '''copies all files at the appropriate location in this mdb'''
        mdb = self.mdb
        if not 'morphology' in mdb.keys():
            print 'copying morphology {}'.format(self.hocpath)
            mdb.create_managed_folder('morphology')
            I.shutil.copy(self.hocpath, mdb['morphology'])
        

        # copying this, because we might have a different morpholgy here compared to the 
        # biophysics fitting, which might be incompatzible with the confile.
        for k in ['fixed_params','get_Combiner','get_Evaluator',
                  'get_Simulator','get_fixed_params','params']:
            mdb[k] = self.biophysical_model_mdb[self.biophysical_model_mdb_key][k]
            
        if not 'network_embedding'  in mdb.keys():
            print 'create network embedding subfolder'
            mdb.create_sub_mdb('network_embedding')     
        for loc in self.locs:
            if not loc in mdb['network_embedding'].keys():
                print 'creating network_embedding subfolder for {}'.format(loc)
                mdb['network_embedding'].create_managed_folder(loc)
               
        def get_file_or_folder_that_startswith(path, startswith):
            paths = [p for p in I.os.listdir(path) if p.startswith(startswith)]
            assert len(paths) == 1
            return I.os.path.join(path,paths[0])
        
        def get_file_or_folder_that_endswith(path, endswith):
            paths = [p for p in I.os.listdir(path) if p.endswith(endswith)]
            assert len(paths) == 1
            return I.os.path.join(path,paths[0])  
        print ''
        print 'creating number of cells spreadsheet from confile'
        from singlecell_input_mapper.singlecell_input_mapper import con_file_to_NumberOfConnectedCells_sheet
        for loc in self.locs:
            d = self.anatomical_model_mdb[self.anatomical_model_mdb_key]
            p = get_file_or_folder_that_startswith(d, loc+'_')
            syn = get_file_or_folder_that_endswith(p, '.syn')
            con = get_file_or_folder_that_endswith(p, '.con')
            I.shutil.copy(syn, mdb['network_embedding'][loc].join('con.syn'))
            I.shutil.copy(con, mdb['network_embedding'][loc].join('con.con'))
            con_file_to_NumberOfConnectedCells_sheet(mdb['network_embedding'][loc].join('con.con'))
            print con 
        
        print ''            
        print 'create evoked activity parameterfiles'
        from getting_started import getting_started_dir
        ongoing_template_param_name = I.os.path.join(getting_started_dir, 'functional_constraints/ongoing_activity/ongoing_activity_celltype_template_exc_conductances_fitted.param')               
        print 'template: {}'.format(ongoing_template_param_name)
        
        for loc in self.locs:
            for stim in self.stims:
                print '\tstim: {}, loc: {}'.format(stim,loc)
                ongoing_template_param_name = I.os.path.join(getting_started_dir, 'functional_constraints/ongoing_activity/ongoing_activity_celltype_template_exc_conductances_fitted.param')
                cell_number_file_name = mdb['network_embedding'][loc].join('NumberOfConnectedCells.csv')
                syn_file_path = mdb['network_embedding'][loc].join('con.syn')
                con_file_path = mdb['network_embedding'][loc].join('con.con')
                cell_number_file_name = mdb['network_embedding'][loc].join('NumberOfConnectedCells.csv')
                with I.silence_stdout:
                    I.create_evoked_network_parameter(ongoing_template_param_name, 
                                          cell_number_file_name, 
                                          syn_file_path, 
                                          con_file_path, 
                                          stim, 
                                          mdb['network_embedding'][loc].join(stim + '_network.param'))
        
        print 
    def get_network_param(self, loc = 'C2center', stim = 'C2', network_param_modfuns = [], stim_onset = 2000):
        network_param = self.mdb['network_embedding'][loc].join('{}_network.param'.format(stim))
        network_param = I.scp.build_parameters(network_param)
        for fun in network_param_modfuns:
            fun(network_param)
        set_stim_onset(network_param, stim_onset)
        return network_param  
    
        
#####################
# model selection
#####################
class ModelSelection:
    ''' Handles selection of models and allows analyzing them'''
    def __init__(self, l6_config, 
                 objectives_BAC = objectives_BAC,
                 objectives_step = objectives_step
    ):
        self.l6_config = l6_config
        self.param_names = l6_config.biophysical_model_mdb[l6_config.biophysical_model_mdb_key]['params'].index
        self.pdf_XY_splitted, self.pdf_XY = self._get_pdfXY()
        self.objectives = [p for p in self.pdf_XY.columns if not p in self.param_names]
        self.objectives_BAC = objectives_BAC
        self.objectives_step = objectives_step 
        self.selected_models = []      
        
    def _get_pdfXY(self):
        from biophysics_fitting.model_selection import get_model_pdf_from_mdb, get_pdf_selected    
        pdf_XY_splitted, pdf_XY = get_model_pdf_from_mdb(
            self.l6_config.biophysical_model_mdb[self.l6_config.biophysical_model_mdb_key])
        return pdf_XY_splitted, pdf_XY
        
    def select_models(self, BAC_limit = 3.5, step_limit = 4.5):
        from biophysics_fitting.model_selection import get_model_pdf_from_mdb, get_pdf_selected            
        selected_models = []
        for k, pdf in self.pdf_XY_splitted.iteritems():
            if not k in self.l6_config.runs: continue
            print self.objectives_BAC
            p, selected_model = get_pdf_selected(pdf, BAC_limit = BAC_limit, step_limit = step_limit,
                                                 objectives_BAC=self.objectives_BAC,
                                                 objectives_step=self.objectives_step)
            selected_models.append(selected_model)
            I.display.display(p)      
        self.selected_models = selected_models  
        return selected_models # get_pdf_selected(pdf, BAC_limit = 3.5, step_limit = 4.5)
    
    def get_parameters(self, modelid):
        return self.pdf_XY.loc[modelid][self.param_names]
    
    def get_objectives(self, modelid, group = 'all'):
        '''group: all, BAC or step'''
        if group == 'all':
            o = self.objectives
        elif group == 'BAC':
            o = self.objectives_BAC
        elif group == 'step':
            o = self.objectives_step
        else:
            raise ValueError()
        return self.pdf_XY.loc[modelid][self.objectives]

    def get_cell_param(self, modelid, add_sim_param = False, recordingSites = [], tStop = 295):
        simulator = self.get_simulator()
        cell_param = simulator.setup.get_cell_params(self.get_parameters(modelid)) 
        if add_sim_param:
            cell_param = _add_sim_to_neuron_param(cell_param, 
                                                  recordingSites = recordingSites, 
                                                  tStop = tStop)
        return cell_param
    
    def get_simulator(self):
        mdb = self.l6_config.mdb # biophysical_model_mdb[self.l6_config.biophysical_model_mdb_key]
        return mdb['get_Simulator'](mdb)
    
    def get_cell_with_biophysics(self, modelid):
        return self.get_simulator().setup.get(self.get_parameters(modelid))
    

##################
# simulate and visualize model responses
###################
class o:
    '''object opaque for dask. Allows passing sumatra.NTParameter structures.
    They don't get serialized propely by dask otherwise'''
    def __init__(self,obj):
        self.obj = obj

def get_ax_ids(ll):
    '''takes list of list specifying the arangement of subplots.
    returns tuples, such that matplotlib axes are arranged in the same way
    
    ll = [['upper left','upper right'],['lower left','lower mid','lower right']]
    get_ax_ids(ll)
    {(2, 2, 1): 'upper left',
     (2, 2, 2): 'upper right',
     (2, 3, 4): 'lower left',
     (2, 3, 5): 'lower mid',
     (2, 3, 6): 'lower right'}
    '''
    return {(len(ll), len(line), linenr*len(line) + colnr + 1): col
            for linenr, line in enumerate(ll)
            for colnr, col in enumerate(line)}
    
         
class ModelResponses:
    def __init__(self, modelSelection):
        self.modelSelection = modelSelection
    
    def _save_cell(self, modelid, stim, cell):
        mdb = self.modelSelection.l6_config.mdb
        mdb = mdb.create_sub_mdb(modelid, raise_ = False)
        mdb = mdb.create_sub_mdb('model_responses', raise_ = False)        
        mdb = mdb.create_sub_mdb(stim, raise_ = False)
        mdb.setitem('cell', cell, dumper = I.dumper_cell)
        mdb.setitem('tVec', I.np.array(cell.tVec), dumper = I.dumper_numpy_to_npz)
        mdb.setitem('vmSoma', I.np.array(cell.soma.recVList[0]), dumper = I.dumper_numpy_to_npz)
        
    def _save_return_helper(self, cell, save, return_cell, modelid, stim):
        if save:
            self._save_cell(modelid, stim, cell)
        if return_cell:
            return cell
        
    @I.dask.delayed
    def simulate_current_stim(self, modelid, stim, return_cell = False, save = True):
        s = self.modelSelection.get_simulator()
        p = self.modelSelection.get_parameters(modelid)
        cell, _ = s.get_simulated_cell(p, stim)
        return self._save_return_helper(cell, save, return_cell, modelid, stim)
    
    @I.dask.delayed
    def simulate_synaptic_stim(self, modelid, network_param_o, 
                               stim_description, return_cell = False, 
                               save = True, tStop = 2500):
        # dask does not serialize sumatra.NTParameterSet properly, therefore
        # we encapsulate it
        network_param = network_param_o.obj 
        cell, _ = self.modelSelection.get_cell_with_biophysics(modelid)
        sim = I.scp.NTParameterSet({'tStart': 0.0, 'tStop': tStop, 'dt': 0.025, 
                                    'Vinit': -75.0, 'dt': 0.025, 'T': 34.0})        
        evokedNW = I.scp.NetworkMapper(cell, network_param.network, sim)
        evokedNW.create_saved_network2()
        I.scp.init_neuron_run(sim, vardt = False)
        return self._save_return_helper(cell, save, return_cell, modelid, stim_description)

    @I.dask.delayed
    def simulate_initialization(self, modelid, return_cell = False, save = True):
        cell, _ = self.modelSelection.get_cell_with_biophysics(modelid)
        sim = I.scp.NTParameterSet({'tStart': 0.0, 'tStop': 300, 'dt': 0.025, 
                                    'Vinit': -75.0, 'dt': 0.025, 'T': 34.0})       
        I.scp.init_neuron_run(sim, vardt = True)
        return self._save_return_helper(cell, save, return_cell, modelid, 'init')
        
    def run_simulation_for_selected_models(self, client):
        selected_models = self.modelSelection.selected_models
        delayeds = [self.simulate_current_stim(modelid, stim) 
                    for modelid in selected_models
                    for stim in self.modelSelection.get_simulator().setup.get_stims()]
        delayeds.extend([self.simulate_initialization(modelid) for modelid in selected_models])
        network_param = self.modelSelection.l6_config.get_network_param(loc = 'C2center', 
                                                                       stim = 'C2', 
                                                                       stim_onset = 2000)
        delayeds.extend([self.simulate_synaptic_stim(modelid, o(network_param), 'PW_stim') 
                         for modelid in selected_models])  
        print len(delayeds)      
        futures = client.compute(delayeds)
        I.distributed.fire_and_forget(futures)
        return delayeds, futures
        
    def _extract_vm_from_cell(self, cell):
        '''helper function to extract membrane potential at 'important' locations'''
        l6_config = self.modelSelection.l6_config
        biophysical_model_mdb = l6_config.biophysical_model_mdb
        biophysical_model_mdb_key = l6_config.biophysical_model_mdb_key
        fixed_params = biophysical_model_mdb[biophysical_model_mdb_key]['fixed_params']
        BAC_dist = fixed_params['BAC.stim.dist']
        hot_zone_dist = fixed_params['hot_zone.max_']*.5 + fixed_params['hot_zone.max_']*.5
        bAP_dist_1 = fixed_params['bAP.hay_measure.recSite1']
        bAP_dist_2 = fixed_params['bAP.hay_measure.recSite2']    
        return {
            'soma': cell.soma.recVList[0],
            'hot_zone': biophysics_fitting.utils.vmApical(cell, hot_zone_dist),
            'bAP1': biophysics_fitting.utils.vmApical(cell, bAP_dist_1),
            'AIS': [s for s in cell.sections if s.label == 'AIS'][1].recVList[-1]
            #'bAP2': biophysics_fitting.utils.vmApical(cell, 529)
            }
        
    def _plot_helper(self, cell, ax):
        colormap = I.defaultdict(lambda: 'grey')
        colormap['hot_zone'] = 'r'
        colormap['soma'] = 'k'
        colormap['AIS'] = 'g'
        for k, v in self._extract_vm_from_cell(cell).iteritems():
            ax.plot(cell.tVec, v, label = k, c = colormap[k], alpha = .5 if k == 'AIS' else 1)        
            
    def visualize_selected_models(self, ax_arrangement = [['PW_stim'],
                          ['BAC', 'bAP', 'init'],['StepOne', 'StepTwo', 'StepThree', ]],
                            xlim = {'BAC': (245,400),
                                    'bAP': (245,400)},
                            ylim = {'init': (-90, -60)}):
        selected_models = self.modelSelection.selected_models
        mdb = self.modelSelection.l6_config.mdb
        for model in selected_models:
            print selected_models
            m = mdb[str(model)]['model_responses']         
            fig = I.plt.figure(figsize = (15*0.7,12*.7), dpi = 200)            
            for ax_position, stim in get_ax_ids(ax_arrangement).iteritems():
                ax = fig.add_subplot(*ax_position)
                self._plot_helper(m[stim]['cell'], ax)
                ax.set_ylim(-90,50)
                ax.set_title(stim)
                if stim in xlim:
                    ax.set_xlim(*xlim[stim])
                if stim in ylim:
                    ax.set_ylim(*ylim[stim])
            I.plt.tight_layout()
            I.sns.despine()
            I.display.display(fig)
            I.plt.close()
            
###################
# synaptic strength fitting
###################

def _add_sim_to_neuron_param(neuron_param, recordingSites = [], tStop = 295):
        NTParameterSet = I.scp.NTParameterSet
        sim_param = {'tStart': 0.0, 'tStop': 295, 'dt': 0.025, 'Vinit': -75.0, 
                     'dt': 0.025, 'T': 34.0, 'recordingSites': recordingSites}
        NMODL_mechanisms = {}
        return NTParameterSet({'neuron': neuron_param, 
                               'sim': sim_param, 
                               'NMODL_mechanisms': NMODL_mechanisms})
        
class SynapticStrengthFitting:
    def __init__(self, model_selection):
        self.model_selection = model_selection
        self.psps = I.defaultdict(dict)
        self._psp_descriptors = []
        self._psps = []
        self.neuron_param = None
        self.confile = None
    
    def run_synaptic_strength_fitting(self, client, loc = 'C2center'):
        mdb = self.model_selection.l6_config.mdb
        for model in self.model_selection.selected_models:
            if 'syn_strength_fitting' in mdb[model].keys():
                print 'skipping model {} as it seems to be simulated already. If the simulation '.format(model)
                'run was incomplete, you can delete the data by running del l6_config.mdb[\'{}\'][\'{}\']'.format(model, 'syn_strength_fitting')                
                continue         
            cell_param = self.model_selection.get_cell_param(model)
            cell_param = _add_sim_to_neuron_param(cell_param)
            confile = mdb['network_embedding'][loc].get_file('.con')
            self.confile = confile
            self.neuron_param = cell_param
            psp = I.simrun3.synaptic_strength_fitting.PSPs(cell_param, confile)
            if client is not None:
                psp.run(client)
            self.psps[model][loc] = psp
        
    def save_psps_to_mdb(self):
        mdb = self.model_selection.l6_config.mdb
        for model, psps in self.psps.iteritems():
            mdb[model].create_sub_mdb('syn_strength_fitting', raise_ = False)
            for loc, psps in psps.iteritems():
                psps.get_voltage_and_timing()
                mdb[model]['syn_strength_fitting'][loc] = psps
    
    def get_psp_from_database(self, modelid, loc):
        mdb = self.model_selection.l6_config.mdb
        psp = mdb[modelid]['syn_strength_fitting'][loc]
        return psp
    
    def get_optimal_g(self, model_id, loc = 'C2center', method = 'dynamic_baseline', threashold = 0.1):
        '''returns dataframe containing optimal synaptic strnghts. Assumes, that all voltage
        traces from synaptic strength fitting are saved in mdb[model_id][syn_strength_fitting_psp]'''
            
        #model_id = selected_models[0]
        psp = self.get_psp_from_database(model_id, loc)
        vt = psp.get_voltage_and_timing(method, merged=True)
        pdf = I.simrun3.synaptic_strength_fitting.ePSP_summary_statistics(vt,  threashold = threashold)
        pdf_linfit = pdf.reset_index()
        pdf_linfit = pdf_linfit.groupby('celltype').apply(I.simrun3.synaptic_strength_fitting.linear_fit_pdf)
        pdf = I.pd.concat([pdf_linfit, I.barrel_cortex.get_EPSP_measurement()], axis = 1)
        I.simrun3.synaptic_strength_fitting.calculate_optimal_g(pdf)        
        return pdf.rename(index = {'VPM_C2': 'VPM'})    
        
#     def visualize_psps(self, model_id, loc = 'C2center', g = 1.0, method = 'dynamic_baseline'):
#         psp = self.get_psp_from_database(model_id, loc)
#         vt = psp.get_voltage_and_timing(method)        
#         #vt = I.simrun3.synaptic_strength_fitting.get_voltage_and_timing(vt, method)
#         vt = I.simrun3.synaptic_strength_fitting.merge_celltypes(vt) 
#         pdf = I.pd.concat([I.pd.Series([x[1] for x in vt[name][g][g]], name = name) 
#              for name in vt.keys()], axis = 1)
#         fig = I.plt.figure(figsize = (10,len(vt)*1.3))
#         ax = fig.add_subplot(111)
#         pdf.plot(kind = 'hist', subplots = True, bins = I.np.arange(0,pdf.max().max(), 0.01), ax = ax)
    def visualize_psps(self, model_id, loc = 'C2center', method = 'dynamic_baseline', g = 1.0):
        psp = self.get_psp_from_database(model_id, loc)
        psp.visualize_psps(method = method, g = g)

                
    def save_synapses_to_landmark_file(self, modelid, population, outfile, loc = 'C2center'):
        with I.silence_stdout:
            cell, _ = self.model_selection.get_cell_with_biophysics(modelid)
        network_param = self.model_selection.l6_config.get_network_param() 
        nwMap = I.scp.NetworkMapper(cell, network_param.network, I.scp.NTParameterSet({'tStop': 10}))
        with I.silence_stdout:
            nwMap._assign_anatomical_synapses()   
            nwMap._create_presyn_cells()    
        syn_coordinates = [s.coordinates for s in cell.synapses[population]]
        I.scp.write_landmark_file(outfile, syn_coordinates)
        return syn_coordinates 

############################################
# write landmark files
############################################

def write_landmark_file(model_selection):
    with I.silence_stdout:
        cell, _ = model_selection.get_cell_with_biophysics(model_selection.selected_models[0])

    fixed_params = model_selection.l6_config.mdb['fixed_params']
    
    landmark_positions = [biophysics_fitting.utils.vmApical_position(cell, dist = d) 
                          for d in [fixed_params['bAP.hay_measure.recSite1'], fixed_params['bAP.hay_measure.recSite2']]] 
    
    I.scp.write_landmark_file(model_selection.l6_config.mdb['morphology']\
                              .join('recSites.landmarkAscii'), landmark_positions)
#############################################
# general method for setting up simulations of evoked activity
#############################################
class EvokedActivitySimulationSetup:
    def __init__(self, output_dir_key = None, synaptic_strength_fitting = None, 
                 stims = None, locs = None, INHscalings = None, ongoing_scale = None, ongoing_scale_pop = None, nProcs = 1, nSweeps = 200, tStim = 245, tEnd = 295,
                 models = None):
        self.output_dir_key = output_dir_key
        self.synaptic_strength_fitting = synaptic_strength_fitting
        self.model_selection = synaptic_strength_fitting.model_selection
        self.l6_config = self.model_selection.l6_config
        self.stims = stims
        self.locs = locs
        self.INHscaling = INHscalings
        self.ongoing_scale = ongoing_scale #rieke
        self.ongoing_scale_pop = ongoing_scale_pop
        self.nProcs = nProcs
        self.nSweeps = nSweeps # /rieke
        self.tStim = tStim
        self.tEnd = tEnd
        self.ds = []
        self.futures = None
        self.models = models
        if self.models is None:
            self.models = self.synaptic_strength_fitting.model_selection.selected_models
            
    def setup(self):
        mdb = self.l6_config.mdb
        
        for model_id in self.models:
            syn_strength = self.synaptic_strength_fitting.get_optimal_g(model_id)['optimal g']
            print 'syn_strength'
            I.display.display(syn_strength)
            if not self.output_dir_key in mdb[str(model_id)].keys():
                mdb[str(model_id)].create_managed_folder(self.output_dir_key)
            else:
                print 'skipping model {} as it seems to be simulated already. If the simulation '.format(model_id)
                'run was incomplete, you can delete the data by running del l6_config.mdb[\'{}\'][\'{}\']'.format(model_id, self.output_dir_key)
                continue
            landmark_name = mdb['morphology'].join('recSites.landmarkAscii')        
            cell_param = self.model_selection.get_cell_param(model_id, add_sim_param = True,
                                                        recordingSites = [landmark_name])
            cell_param_name = mdb[str(model_id)]['PW_fitting'].join('cell.param')
            cell_param.save(cell_param_name)
            for INH_scaling in self.INHscaling:            
                for stim in self.stims:
                    for loc in self.locs:
                        network_param = self.l6_config.get_network_param(stim = stim, 
                                                                    loc = loc, 
                                                                    stim_onset=self.tStim)
                        I.scp.network_param_modify_functions.change_evoked_INH_scaling(network_param, INH_scaling)
                        I.scp.network_param_modify_functions.change_glutamate_syn_weights(network_param, syn_strength)
                        I.scp.network_param_modify_functions.change_ongoing_interval(network_param, factor = self.ongoing_scale, pop = self.ongoing_scale_pop) ##adjust ongoing activity if necessary
                        network_param_name = mdb[str(model_id)][self.output_dir_key].join('network_INH_{}_stim_{}_loc_{}.param'.format(INH_scaling, stim, loc))
                        network_param.save(network_param_name)
                        outdir = mdb[str(model_id)][self.output_dir_key].join(str(INH_scaling)).join(stim).join(str(loc))
                        print model_id, INH_scaling, stim, loc
                        d = I.simrun_run_new_simulations(cell_param_name, network_param_name, 
                                                         dirPrefix = outdir, 
                                                         nSweeps = self.nSweeps, 
                                                         nprocs = self.nProcs, 
                                                         scale_apical = lambda x: x,
                                                         silent = False,
                                                         tStop = self.tEnd)
                        self.ds.append(d)

    def run(self, client, fire_and_forget = False):
        if len(self.ds) == 0:
            raise RuntimeError("You must run the setup method first")
        self.futures = client.compute(self.ds)
        if fire_and_forget:
            I.distributed.fire_and_forget(self.futures)
            
    def init_mdb(self, client, dendritic_voltage_traces = False):
        mdb = self.l6_config.mdb
        for model_id in self.model_selection.selected_models:
            mdb_key = self.output_dir_key + '_mdb'
            if mdb_key in mdb[str(model_id)].keys():
                del mdb[str(model_id)][mdb_key]
            if not mdb_key in mdb[str(model_id)].keys():
                mdb_init = mdb[str(model_id)].create_sub_mdb(mdb_key)
                I.mdb_init_simrun_general.init(mdb_init, mdb[str(model_id)][self.output_dir_key], 
                                               burst_times = False, 
                                               get = client.get,
                                               dendritic_voltage_traces = dendritic_voltage_traces)        
            

############
# PW fitting
############

def signchange(x,y):
    if x / abs(x) == y / abs(y):
        return False
    else:
        return True
assert(signchange(-2,1))
assert(~signchange(2,1))
assert(~signchange(-22,-1))
assert(signchange(22,-1789))

def linear_interpolation_between_pairs(X,Y, x):
    assert(x<=max(X))
    assert(x>=min(X))
    pair = [lv for lv in range(len(X)-1) if signchange(X[lv]-x, X[lv+1]-x)]
    assert(len(pair) == 1)
    pair = pair[0]
    m = (Y[pair+1]-Y[pair]) / (X[pair+1]-X[pair])
    c = Y[pair]-X[pair]*m
    return m*x+c    

class PWfitting:
    def __init__(self, l6_config, model_selection, min_time = 8, max_time = 25, mdb_path1 = '/nas1/Data_arco/results/20190114_spiketimes_database', mdb_path2 = '/nas1/Data_arco/results/mdb_robert_3x3/', stim = 'D2', cellid = None):
        self.l6_config = l6_config
        self.model_selection = model_selection
        self.min_time = min_time
        self.max_time = max_time
        self.mdb_path1 = mdb_path1
        self.mdb_path2 = mdb_path2
        self.stim = stim
        if cellid == None:
            self.cellid = l6_config.biophysical_model_mdb_key
        else:
            self.cellid = cellid
        ### get target value
        #st_CDK = I.ModelDataBase('/nas1/Data_arco/results/20190114_spiketimes_database')['CDK_PassiveTouch'] # rieke
        st_CDK = I.ModelDataBase(self.mdb_path1)['CDK_PassiveTouch']
        
        st_CDK = I.select(st_CDK, stim = self.stim) #changed by rieke
        self.CDK_target_value_avg_L5tt = I.temporal_binning(st_CDK, min_time = min_time, 
                                                       max_time = max_time, 
                                                       bin_size = max_time-min_time)[1][0]
        self.CDK_target_value_same_cell = I.temporal_binning(I.select(st_CDK, cell = self.cellid), 
                                                      min_time = min_time, 
                                                             max_time = max_time, 
                                                             bin_size = max_time-min_time)[1][0]

        print 'CDK_target_value_avg_L5tt: %s' % self.CDK_target_value_avg_L5tt
        print 'CDK_target_value_same_cell: %s' % self.CDK_target_value_same_cell
        
        #st_robert_control = I.ModelDataBase('/nas1/Data_arco/results/mdb_robert_3x3/')['spike_times']
        st_robert_control = I.ModelDataBase(self.mdb_path2)['spike_times']
        
        st_robert_control = st_robert_control[st_robert_control.index.str.split('_').str[0] == 'C2']
        self.target_robert = I.temporal_binning(st_robert_control, min_time = 245+min_time, 
                                                max_time = 245+max_time, 
                                                bin_size = max_time-min_time)[1][0]
        print 'target_robert: %s' % self.target_robert

    def _get_INH_dependent_n_spikes(self, model):
        st = self.l6_config.mdb[str(model)]['PW_fitting_mdb']['spike_times']
        st['INH'] = st.index.str.split('/').str[0]
        fun = lambda x: I.temporal_binning(x, 
                                           min_time = 245+self.min_time, 
                                           max_time = 245+self.max_time, 
                                           bin_size = self.max_time - self.min_time)[1][0]
        s = st.groupby('INH').apply(fun)
        s.name = model
        return s
    
    def plot_INH_pspike_relationship(self):


        colors = ['k', 'r', 'b', 'g', 'orange', 'pink']
        model_color = [[model, colors[lv]] 
                       for lv, model 
                       in enumerate(self.model_selection.selected_models)]
        for m, c in model_color:
            s = self._get_INH_dependent_n_spikes(m)
            I.plt.plot(s.index, s, 'o', c = c, label = str(m))
        I.plt.axhline(self.CDK_target_value_avg_L5tt, color = 'k')
        I.plt.axhline(self.CDK_target_value_same_cell)
        I.plt.axhline(self.target_robert)
        I.plt.xticks(s.index.astype('f'))
        I.plt.xlabel('INH')
        I.plt.ylabel('spikes per trial 0-25ms poststim')
        I.plt.legend()
        I.sns.despine()
        
    def get_INH_scaling_dict(self):
        INH_scaling_dict = {}
        for model in self.model_selection.selected_models:
            try:
                s = self._get_INH_dependent_n_spikes(model)
                INH_scaling_dict[model] = linear_interpolation_between_pairs(s.astype('float'), 
                                                                             s.index.astype('float'), 
                                                                             self.CDK_target_value_avg_L5tt)
            except AssertionError:
                pass
        return INH_scaling_dict
    
    def _plot_PSTHs(self, INH, CDK_PW = True, robert = True, models = True, plottype = 'hist'):
            mdb = self.l6_config.mdb
            if plottype == 'hist':
                plotfun = I.histogram
            elif plottype == 'line':
                def plotfun(bins, label = None, fig = None, colormap = None):
                    fig.plot(bins[0][:-1], bins[1], color = colormap[label], label = label)
            st_CDK = I.ModelDataBase(self.mdb_path1)['CDK_PassiveTouch']
            st_CDK = I.select(st_CDK, stim = 'D2')
            CDK_bins = I.temporal_binning(st_CDK, min_time = -144, max_time = 100, bin_size = 1)
            CDK_bins = [range(245-144,245+100+1), CDK_bins[1]]
            st_robert_control = I.ModelDataBase(self.mdb_path2)['spike_times']
            st_robert_control = st_robert_control[st_robert_control.index.str.split('_').str[0] == 'C2']
            robert_bins = I.temporal_binning(st_robert_control, min_time = 0, max_time = 245+50, bin_size = 1)
            cmap = I.defaultdict(lambda: None)
            cmap['CDK_PW'] = 'grey'
            cmap['robert'] = 'blue'
            fig = I.plt.figure(figsize = (15,3))
            if CDK_PW: plotfun(CDK_bins,label = 'CDK_PW', fig = fig.add_subplot(111), colormap = cmap)   
            if robert: plotfun(robert_bins,label = 'robert', fig = fig.add_subplot(111), colormap = cmap)    

            if models:
                for model in self.model_selection.selected_models:  
                    st = mdb[str(model)]['PW_fitting_mdb']['spike_times']
                    st = st[st.index.str.split('/').str[0] == str(INH)]
                    plotfun(I.temporal_binning(st, min_time = 0, max_time = 350), 
                                fig = fig.add_subplot(111), 
                                label = str(model), colormap = cmap)
            I.plt.ylim([0,0.3])        
            I.plt.title(str(INH))
            I.plt.legend()
            return fig
    def plot_PSTHs(self):
        for INH in I.np.arange(0.5,2.1,.1):
            self._plot_PSTHs(INH, CDK_PW = True, robert = True)
   
def get_all_L6_simulation_dbs():
    mdb = I.ModelDataBase('/nas1/Data_arco/results/20190507_metaanalysis_of_L6_control_experiments')
    # create mdb dict
    mdbs = {}
    for k in mdb.keys():
        if k == 'local': continue
        m = mdb[k]
        l6_config = m['l6_config']
        selected_models = m['selected_models']
        for model in selected_models:
            model_run = model.split('_')[-3]
            morphology = k.split('_')[4]        
            try:
                m = l6_config.mdb[model]['3x3_control_mdb']
            except:
                print 'skipping ',morphology, model_run
                continue
            mdbs[str(model_run) + '_' + str(morphology)] = m
    return mdbs
#####
# unused so far

# def setup_cell_synaptic_input(cell, network_param, sim_param):
#     evokedNW = I.scp.NetworkMapper(cell, network_param.network, sim_param)
#     evokedNW.create_saved_network2()
#     return evokedNW
# 
# def setup_cell_synaptic_input_PW(cell, param_modfuns = []):
#     from sumatra.parameters import NTParameterSet
#     network_param = get_network_param()
#     for fun in param_modfuns:
#         fun(network_param)
#     sim_param = NTParameterSet(dict(tStop = 2500))
#     setup_cell_synaptic_input(cell, network_param, sim_param)
#     
# def get_param_by_modelid(mdb, modelid):
#     param_names = mdb['params'].index
#     return pdf_XY.loc[modelid][param_names]
# 
# def get_cell_with_biophysics(m, modelid):
#     s = m['get_Simulator'](m)
#     param = get_param_by_modelid(m, modelid)
#     cell, param = s.setup.get(param)
#     return cell   
# 
# @I.dask.delayed
# def simulate(mdb, modelid, savemdb = None, cell_setup_fun = lambda x: x, 
#              tStop = 300, vardt = False, return_cell = False, cell_param_mod_fun = None): 
#     cell = get_cell_with_biophysics(mdb, modelid)
#     if cell_param_mod_fun is not None:
#         raise NotImplementedError()
#         cell_param_mod_fun(cell_param)
#     cell_setup_fun(cell)
#     sim = I.scp.NTParameterSet({'tStart': 0.0, 'tStop': tStop, 'dt': 0.025, 'Vinit': -75.0, 'dt': 0.025, 'T': 34.0})
#     I.scp.init_neuron_run(sim, vardt = vardt)
#     if savemdb is not None:
#         savemdb.setitem('cell', cell, dumper = I.dumper_cell)
#         savemdb.setitem('tVec', I.np.array(cell.tVec), dumper = I.dumper_numpy_to_npz)
#         savemdb.setitem('vmSoma', I.np.array(cell.soma.recVList[0]), dumper = I.dumper_numpy_to_npz)
#     if return_cell:
#         return cell
#     else:
#         return I.np.array(cell.tVec), I.np.array(cell.soma.recVList[0])
#   

