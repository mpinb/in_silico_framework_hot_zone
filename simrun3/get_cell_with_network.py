import Interface as  I

def get_cell_with_network(neuron_param, network_param):
    '''This returns a function, that returns a freshly initialized cell and network.
    It takes advantage of the re_init methods, i.e. it is more efficient than 
    recreating cell and network from scratch.'''
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