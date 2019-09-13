import Interface as  I
import warnings
def get_cell_with_network(neuron_param, network_param, cache = True):
    '''This returns a function, that returns a freshly initialized cell and network.
    It takes advantage of the re_init methods, i.e. it is more efficient than 
    recreating cell and network from scratch.'''
    if not cache:
        return get_cell_with_network_without_cache(neuron_param, network_param)
    try:
        neuron_param['neuron']['cell_modify_functions']['synaptic_input']
        errstr = "Found cell_modify_function of type synaptic_input. "
        errstr += "This is incompatible with cell caching. Caching "
        errstr += "is therefore switched of. Please note: runtime will "
        errstr += "increase."
        warnings.warn(errstr)
        return get_cell_with_network_without_cache(neuron_param, network_param)
    except KeyError:
        pass
    
    
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
        return cell,nwMap
    return fun

def get_cell_with_network_without_cache(neuron_param, network_param):
    '''This returns a function, that returns a freshly initialized cell and network.
    It takes advantage of the re_init methods, i.e. it is more efficient than 
    recreating cell and network from scratch.'''
    def fun():
        cell = I.scp.create_cell(neuron_param.neuron)  
        nwMap = I.scp.NetworkMapper(cell, network_param.network)
        nwMap.create_saved_network2()
        return cell, nwMap
    return fun