# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import functools
import copy
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.architecture import actvn as actvn

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 subarch='n_layers', num_D=3, getIntermFeat=False, stride=2, gpu_ids=[]):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.getIntermFeat = getIntermFeat
        self.subarch = subarch
     
        for i in range(num_D):            
            netD = self.create_singleD(opt, subarch, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride)
            setattr(self, 'discriminator_%d' % i, netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def create_singleD(self, opt, subarch, input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride):
        if subarch == 'adaptive':
            netD = AdaptiveDiscriminator(opt, input_nc, ndf, n_layers, norm_layer, getIntermFeat, opt.adaptive_D_layers)        
        elif subarch == 'n_layers':
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, getIntermFeat, stride)
        else:
            raise ValueError('unrecognized discriminator sub architecture %s' % subarch)
        return netD

    # should return list of outputs
    def singleD_forward(self, model, input, ref):
        if self.subarch == 'adaptive':
            return model(input, ref)
        elif self.getIntermFeat:
            return model(input)
        else:
            return [model(input)]

    # should return list of list of outputs
    def forward(self, input, ref=None):        
        result = []
        input_downsampled = input
        ref_downsampled = ref
        for i in range(self.num_D):
            model = getattr(self, 'discriminator_%d' % i)
            result.append(self.singleD_forward(model, input_downsampled, ref_downsampled))            
            input_downsampled = self.downsample(input_downsampled)
            ref_downsampled = self.downsample(ref_downsampled) if ref is not None else None
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):    
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False, stride=2):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers        

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))        
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=stride, padding=padw), nn.LeakyReLU(0.2, False)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            item = [
                norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                nn.LeakyReLU(0.2, False)
            ]
            sequence += [item]
                
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2, False)
        ]]
        
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        
        for n in range(len(sequence)):                        
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))        

    def forward(self, input):
        res = [input]
        for n in range(self.n_layers + 2):            
            model = getattr(self, 'model'+str(n))
            x = model(res[-1])            
            res.append(x)
        if self.getIntermFeat:
            return res[1:]
        else:
            return res[-1]

class AdaptiveDiscriminator(BaseNetwork):
    def __init__(self, opt, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, getIntermFeat=False, adaptive_layers=1):
        super(AdaptiveDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.adaptive_layers = adaptive_layers
        self.input_nc = input_nc
        self.ndf = ndf
        self.kw = kw = 4
        self.padw = padw = int(np.ceil((kw-1.0)/2))  
        self.actvn = actvn = nn.LeakyReLU(0.2, True)
                
        self.sw = opt.fineSize // 8
        self.sh = int(self.sw / opt.aspect_ratio)
        self.ch = self.sh * self.sw        

        nf = ndf        
        self.fc_0 = nn.Linear(self.ch, input_nc*(kw**2))
        self.encoder_0 = nn.Sequential(*[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), actvn])
        for n in range(1, self.adaptive_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)             
            setattr(self, 'fc_'+str(n), nn.Linear(self.ch, nf_prev*(kw**2)))            
            setattr(self, 'encoder_'+str(n), nn.Sequential(*[(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)), actvn]))

        sequence = []
        nf = ndf * (2**(self.adaptive_layers-1))
        for n in range(self.adaptive_layers, n_layers+1):
            nf_prev = nf
            nf = min(nf * 2, 512) 
            stride = 2 if n != n_layers else 1
            item = [norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)), actvn]
            sequence += [item]                
        
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            setattr(self, 'model'+str(n + self.adaptive_layers), nn.Sequential(*sequence[n]))

    def gen_conv_weights(self, encoded_ref):
        models = []
        b = encoded_ref[0].size()[0]
        nf = self.ndf
        actvn = self.actvn                
        weight = self.fc_0(nn.AdaptiveAvgPool2d((self.sh, self.sw))(encoded_ref[0]).view(b*nf, -1))        
        weight = weight.view(b, nf, self.input_nc, self.kw, self.kw)
        model0 = []
        for i in range(b):
            model0.append(self.ConvN(functools.partial(F.conv2d, weight=weight[i], stride=2, padding=self.padw), 
                nn.InstanceNorm2d(nf), actvn))
        
        models.append(model0)
        for n in range(1, self.adaptive_layers):            
            ch = encoded_ref[n].size()[1]
            x = nn.AdaptiveAvgPool2d((self.sh, self.sw))(encoded_ref[n]).view(b*ch, -1)
            weight = getattr(self, 'fc_'+str(n))(x)
            
            nf_prev = nf
            nf = min(nf * 2, 512) 
            weight = weight.view(b, nf, nf_prev, self.kw, self.kw)
            model = []
            for i in range(b):
                model.append(self.ConvN(functools.partial(F.conv2d, weight=weight[i], stride=2, padding=self.padw), 
                    nn.InstanceNorm2d(nf), actvn))
            
            models.append(model)
        return models

    class ConvN(nn.Module):
        def __init__(self, conv, norm, actvn):
            super().__init__()                
            self.conv = conv
            self.norm = norm
            self.actvn = actvn

        def forward(self, x):
            x = self.conv(x)            
            out = self.norm(x)
            out = self.actvn(out)
            return out

    def encode(self, ref):
        encoded_ref = [ref]
        for n in range(self.adaptive_layers):
            encoded_ref.append(getattr(self, 'encoder_'+str(n))(encoded_ref[-1]))        
        return encoded_ref[1:]

    def batch_conv(self, conv, x):        
        y = conv[0](x[0:1])
        for i in range(1, x.size()[0]):
            yi = conv[i](x[i:i+1])
            y = torch.cat([y, yi])
        return y

    def forward(self, input, ref):
        encoded_ref = self.encode(ref)
        models = self.gen_conv_weights(encoded_ref)
        res = [input]
        for n in range(self.n_layers+2):            
            if n < self.adaptive_layers:
                res.append(self.batch_conv(models[n], res[-1]))
            else:                
                res.append(getattr(self, 'model'+str(n))(res[-1]))        
        if self.getIntermFeat:
            return res[1:]
        else:
            return res[-1]
