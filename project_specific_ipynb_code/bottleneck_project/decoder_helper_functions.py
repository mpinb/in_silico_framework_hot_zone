import torch
import Interface as I
from tqdm import tqdm
from sklearn.neighbors import KernelDensity
import numpy as np

def forward(self, X_ISI_MCM_list):
        '''
        X_ISI_list: a list containing two elements.
           First element is X, i.e. the same as in My_model_bottleneck
           Second element is ISI for each trial
        '''
        X,ISI_SOMA,ISI_DEND = X_ISI_MCM_list
        assert(isinstance(X,torch.Tensor))
        out = self.linear1(X)
        list_ = [out]
        if self.bottleneck_ISI_soma:
            list_ = list_ + [ISI_SOMA]
        if self.bottleneck_ISI_dend:
            list_ = list_ + [ISI_DEND]            
        out = torch.cat(list_, axis = 1)
        out = self.bottleneck_layer(out)
        for layer in self.layers_after_bottleneck:
            out = self.relu(out)
            out = layer(out)
        out = self.output_layer(out)
        return out

def forward_bottleneck(self, X_ISI_MCM_list):
        '''
        X_ISI_list: a list containing two elements.
           First element is X, i.e. the same as in My_model_bottleneck
           Second element is ISI for each trial
        '''
        X,ISI_SOMA,ISI_DEND = X_ISI_MCM_list
        assert(isinstance(X,torch.Tensor))
        out = self.linear1(X)
        list_ = [out]
        if self.bottleneck_ISI_soma:
            list_ = list_ + [ISI_SOMA]
        if self.bottleneck_ISI_dend:
            list_ = list_ + [ISI_DEND]            
        out = torch.cat(list_, axis = 1)
        return out
        # out = self.bottleneck_layer(out)
        # for layer in self.layers_after_bottleneck:
        #     out = self.relu(out)
        #     out = layer(out)
        # out = self.output_layer(out)
        # return out

def forward_decoder(self, bottleneck):
        '''
        X_ISI_list: a list containing two elements.
           First element is X, i.e. the same as in My_model_bottleneck
           Second element is ISI for each trial
        '''
                    
        #out = torch.cat(bottleneck, axis = 1)
        out = bottleneck
        out = self.bottleneck_layer(out)
        for layer in self.layers_after_bottleneck:
            out = self.relu(out)
            out = layer(out)
        out = self.output_layer(out)
        return out

def get_decoder_info(SA, ISI_SOMA, ISI_DEND, AP_SOMA, VT_SOMA, AP_DEND, VT_DEND, temporal_window_width, model):
    """
    SA: synaptic input array: n_trials x n_cell_types x _n_spatial_bins x n_temporal_bins
    """
    model_outs = []
    bottleneck_outs = []
    is_ = []
    soma_isis = []
    dend_isis = []
    end_t = 60  # last starting t of sliding time_window
    for i in tqdm(range(end_t), desc=f"Sliding {temporal_window_width} ms wide time window from 0 to {end_t}"):  # change range to 60 or 20
        # Iterate over all 80ms-wide intervals from 0 to 60, such that the total included time windows are 0->80 until 60->140ms
        X_ISI_MCM_list = [SA[:,:,:,i:temporal_window_width+i].flatten().view(len(ISI_SOMA),-1).float(),
                        ISI_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1).float(),
                        ISI_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1).float() + 100]
        
        # desired_outputs = [AP_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1),
        #                 VT_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1),
        #                 AP_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1),
        #                 VT_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1)] 
        model_out = forward(model, X_ISI_MCM_list)
        model_out = torch.sigmoid(model_out)
        model_out = model_out.cpu().detach().numpy()
        model_outs.append(model_out)
        bottleneck_out = forward_bottleneck(model, X_ISI_MCM_list).cpu().detach().numpy()
        bottleneck_outs.append(bottleneck_out)
        is_.extend([i]*len(SA))
        soma_isis.append(X_ISI_MCM_list[1])
        dend_isis.append(X_ISI_MCM_list[2])
    bottleneck_out = I.np.concatenate(bottleneck_outs)
    model_out = I.np.concatenate(model_outs)
    soma_isi = I.np.concatenate(soma_isis)
    dend_isi = I.np.concatenate(dend_isis)
    is_ = I.np.array(is_)

    return soma_isi, dend_isi, bottleneck_out, model_out

def get_decoder_info_as_df(SA, ISI_SOMA, ISI_DEND, AP_SOMA, VT_SOMA, AP_DEND, VT_DEND, temporal_window_width, model):
    model_outs = []
    bottleneck_outs = []
    #is_ = []
    soma_isis = []
    dend_isis = []
    for i in tqdm(range(60)):  # change range to 60 or 20
        # Iterate over all 80ms-wide intervals from 0 to 60, such that the total included time windows are 0->80 until 60->140ms
        X_ISI_MCM_list = [SA[:,:,:,i:temporal_window_width+i].flatten().view(len(ISI_SOMA),-1).float(),
                        ISI_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1).float(),
                        ISI_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1).float() + 100]
        model_out = forward(model, X_ISI_MCM_list)
        model_out = torch.sigmoid(model_out)
        model_out = model_out.cpu().detach().numpy()
        model_outs.append(model_out)
        bottleneck_out = forward_bottleneck(model, X_ISI_MCM_list).cpu().detach().numpy()
        bottleneck_outs.append(bottleneck_out)
        #is_.extend([i]*len(SA))
        soma_isis.append(X_ISI_MCM_list[1])
        dend_isis.append(X_ISI_MCM_list[2])
    bottleneck_out = I.np.concatenate(bottleneck_outs)
    model_out = I.np.concatenate(model_outs)
    soma_isi = I.np.concatenate(soma_isis)
    dend_isi = I.np.concatenate(dend_isis)
    #is_ = I.np.array(is_)

    return I.pd.DataFrame.from_dict({"soma_isi": I.np.concatenate(soma_isi), "dend_isi": I.np.concatenate(dend_isi),
    "model_out": I.np.concatenate(model_out), 
    "bottleneck_out_0": bottleneck_out[:,0], "bottleneck_out_1": bottleneck_out[:,1], "bottleneck_out_2": bottleneck_out[:,2]})

def cartesian_product(*arrays):
    import numpy
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)