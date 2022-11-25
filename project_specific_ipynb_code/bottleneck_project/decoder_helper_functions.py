import torch

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