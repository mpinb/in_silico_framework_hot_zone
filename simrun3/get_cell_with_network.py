import Interface as  I
def get_cell_with_network(neuron_param, network_param, cache = True):
    @I.cache
    def fun2():
        cell = I.scp.create_cell(neuron_param.neuron)  
        nwMap = I.scp.NetworkMapper(cell, network_param.network)
        nwMap.create_saved_network2()
        return cell, nwMap
    def fun():
        cell, nwMap = fun2()
        cell.re_init_cell()
        nwMap.re_init_network()
        for sec in cell.sections:
            sec._re_init_vm_recording()
            sec._re_init_range_var_recording()      
        # special case: a second evokedNW exists in cell.evokedNW
        try:
            cell.evokedNW.re_init_network()
            errstr = "Found cell_modify_function of type synaptic_input. "
            errstr += "Explicitly resetting cell.evokedNW."
            errstr += "Please note: this is experimental. Please check the results."
            print errstr
            par = neuron_param['neuron']['cell_modify_functions']['synaptic_input']
            # reactivate synapses
            try:
                # according to synapse activation file if it is specified in the neuron
                synapse_activation_file = par['synapse_activation_file']
                cell.evokedNW.reconnect_saved_synapses(synapse_activation_file)
            except KeyError:
                # according to network statistics if synapse acitvation file is not specified
                cell.evokedNW.create_saved_network2()
        except: # if cell.evokedNW does not exist:
            pass # do nothing
        nwMap.create_saved_network2()
        return cell,nwMap
    return fun