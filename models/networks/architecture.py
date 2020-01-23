# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.normalization import SynchronizedBatchNorm2d

from models.networks.base_network import BaseNetwork, batch_conv
from models.networks.normalization import SPADE
import torch.nn.utils.spectral_norm as sn

def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

def generalConv(adaptive=False, transpose=False):
    class NormalConv2d(nn.Conv2d):
        def __init__(self, *args, **kwargs):        
            super(NormalConv2d, self).__init__(*args, **kwargs)            
        def forward(self, input, weight=None, bias=None, stride=1):            
            return super(NormalConv2d, self).forward(input)
    class NormalConvTranspose2d(nn.ConvTranspose2d):
        def __init__(self, *args, output_padding=1, **kwargs):
            #kwargs['output_padding'] = 1
            super(NormalConvTranspose2d, self).__init__(*args, **kwargs)            
        def forward(self, input, weight=None, bias=None, stride=1):            
            return super(NormalConvTranspose2d, self).forward(input)            
    class AdaptiveConv2d(nn.Module):
        def __init__(self, *args, **kwargs):        
            super().__init__()
        def forward(self, input, weight=None, bias=None, stride=1):            
            return batch_conv(input, weight, bias, stride)
    
    if adaptive: return AdaptiveConv2d
    return NormalConv2d if not transpose else NormalConvTranspose2d   

def generalNorm(norm):
    if 'spade' in norm: return SPADE
    def get_norm(norm):
        if 'instance' in norm:
            return nn.InstanceNorm2d
        elif 'syncbatch' in norm:
            return SynchronizedBatchNorm2d
        elif 'batch' in norm:
            return nn.BatchNorm2d
    norm = get_norm(norm)
    class NormalNorm(norm):
        def __init__(self, *args, hidden_nc=0, norm='', ks=1, params_free=False, **kwargs):
            super(NormalNorm, self).__init__(*args, **kwargs)            
        def forward(self, input, label=None, weight=None):
            return super(NormalNorm, self).forward(input)
    return NormalNorm    

class SPADEConv2d(nn.Module):
    def __init__(self, fin, fout, norm='batch', hidden_nc=0, kernel_size=3, padding=1, stride=1):
        super().__init__()             
        self.conv = sn(nn.Conv2d(fin, fout, kernel_size=kernel_size, stride=stride, padding=padding))        

        Norm = generalNorm(norm)
        self.bn = Norm(fout, hidden_nc=hidden_nc, norm=norm, ks=3)        

    def forward(self, x, label=None):      
        x = self.conv(x)
        out = self.bn(x, label)        
        out = actvn(out)
        return out

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm='batch', hidden_nc=0, conv_ks=3, spade_ks=1, stride=1, conv_params_free=False, norm_params_free=False):
        super().__init__()        
        fhidden = min(fin, fout)
        self.learned_shortcut = (fin != fout)        
        self.stride = stride        
        Conv2d = generalConv(adaptive=conv_params_free)        
        sn_ = sn if not conv_params_free else lambda x: x

        # Submodules
        self.conv_0 = sn_(Conv2d(fin, fhidden, conv_ks, stride=stride, padding=1))
        self.conv_1 = sn_(Conv2d(fhidden, fout, conv_ks, padding=1))
        if self.learned_shortcut:
            self.conv_s = sn_(Conv2d(fin, fout, 1, stride=stride, bias=False))

        Norm = generalNorm(norm)        
        self.bn_0 = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=spade_ks, params_free=norm_params_free)
        self.bn_1 = Norm(fhidden, hidden_nc=hidden_nc, norm=norm, ks=spade_ks, params_free=norm_params_free)
        if self.learned_shortcut:
            self.bn_s = Norm(fin, hidden_nc=hidden_nc, norm=norm, ks=spade_ks, params_free=norm_params_free)        

    def forward(self, x, label=None, conv_weights=[], norm_weights=[]):
        if not conv_weights: conv_weights = [None]*3
        if not norm_weights: norm_weights = [None]*3        
        x_s = self._shortcut(x, label, conv_weights[2], norm_weights[2])        
        dx = self.conv_0(actvn(self.bn_0(x, label, norm_weights[0])), conv_weights[0])
        dx = self.conv_1(actvn(self.bn_1(dx, label, norm_weights[1])), conv_weights[1])
        out = x_s + 1.0*dx
        return out

    def _shortcut(self, x, label, conv_weights, norm_weights):
        if self.learned_shortcut:
            x_s = self.conv_s(self.bn_s(x, label, norm_weights), conv_weights)
        elif self.stride != 1:            
            x_s = nn.AvgPool2d(3, stride=2, padding=1)(x)
        else:
            x_s = x
        return x_s