from biophysics_fitting.setup_stim import setup_soma_step
import single_cell_parser as scp
def synaptic_input(cell, network_param = None, synapse_activation_file = None, tStop = None):
    net = scp.build_parameters(network_param)
    sim = scp.NTParameterSet({'tStop': tStop})
    evokedNW = scp.NetworkMapper(cell, net.network, sim)
    if synapse_activation_file is None:
        print 'activating synapses'
        evokedNW.create_saved_network2()
    else:
        evokedNW.reconnect_saved_synapses(synapse_activation_file)

    cell.evokedNW = evokedNW
    return cell