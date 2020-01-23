# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as sn

from models.networks.base_network import batch_conv
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d

class SPADE(nn.Module):
    def __init__(self, norm_nc, hidden_nc=0, norm='batch', ks=3, params_free=False):
        super().__init__()
        pw = ks//2
        if not isinstance(hidden_nc, list): hidden_nc = [hidden_nc]
        for i, nhidden in enumerate(hidden_nc):                                    
            mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
            
            if not params_free or (i != 0):
                s = str(i+1) if i > 0 else ''                                
                setattr(self, 'mlp_gamma%s' % s, mlp_gamma)
                setattr(self, 'mlp_beta%s' % s, mlp_beta)

        if 'batch' in norm:
            self.batch_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
        else:
            self.batch_norm = nn.InstanceNorm2d(norm_nc, affine=False)

    def forward(self, x, maps, weights=None):
        if not isinstance(maps, list): maps = [maps]
        out = self.batch_norm(x)
        for i in range(len(maps)):
            if maps[i] is None: continue
            m = F.interpolate(maps[i], size=x.size()[2:], mode='bilinear')
            if weights is None or (i != 0):
                s = str(i+1) if i > 0 else ''                                  
                gamma = getattr(self, 'mlp_gamma%s' % s)(m)
                beta = getattr(self, 'mlp_beta%s' % s)(m)
            else:
                j = min(i, len(weights[0])-1)
                gamma = batch_conv(m, weights[0][j])
                beta = batch_conv(m, weights[1][j])
            out = out * (1 + gamma) + beta                                  
        return out
        
def get_nonspade_norm_layer(opt, norm_type='instance'):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = sn(layer)
            subnorm_type = norm_type[len('spectral'):]

        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)

        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'syncbatch':
            norm_layer = SynchronizedBatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer