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

def get_decoder_info(SA, ISI_SOMA, ISI_DEND, temporal_window_width, AP_SOMA, VT_SOMA, AP_DEND, VT_DEND, model):
    model_outs = []   # accumilating all the model outs for all the time points starting from 165 --> 245 and end in 225 --> 305 ms in the simulation (60ms duration/ 60*5000 trials = 300,000 dots) 
    bottleneck_outs = []
    is_ = []
    soma_isis = []
    for i in tqdm(range(60)):  # change range to 60 or 20 
        X_ISI_MCM_list = [SA[:,:,:,i:temporal_window_width+i].flatten().view(len(ISI_SOMA),-1).float(),    # dinuka - ind is not defined before !!! since it has to be the same number of trails accordingly i changed it to the len(ISI_SOMA) = 5000 now 
                        ISI_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1).float(),
                        ISI_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1).float() + 100]
        
        # soma and dend ISI are large
        #X_ISI_MCM_list = [SA[:,:,:,i:temporal_window_width+i].flatten().view(len(ind),-1).float(),
        #                              1000*torch.ones(len(SA),1).float(),
        #                              1000*torch.ones(len(SA),1).float()]
        
        

        desired_outputs = [AP_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1),
                        VT_SOMA[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1),
                        AP_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1),
                        VT_DEND[:,[temporal_window_width+i]].view(len(ISI_SOMA),-1)] 
        model_out = forward(model, X_ISI_MCM_list)
        model_out = torch.sigmoid(model_out)
        model_out = model_out.cpu().detach().numpy()
        model_outs.append(model_out)
        bottleneck_out = forward_bottleneck(model, X_ISI_MCM_list).cpu().detach().numpy()
        bottleneck_outs.append(bottleneck_out)
        is_.extend([i]*len(SA))
        soma_isis.append(X_ISI_MCM_list[1])
    bottleneck_out = I.np.concatenate(bottleneck_outs)
    model_out = I.np.concatenate(model_outs)
    soma_isi = I.np.concatenate(soma_isis)
    is_ = I.np.array(is_)

    return soma_isi, bottleneck_out, model_out

def cartesian_product(*arrays):
    import numpy
    la = len(arrays)
    dtype = numpy.result_type(*arrays)
    arr = numpy.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(numpy.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)