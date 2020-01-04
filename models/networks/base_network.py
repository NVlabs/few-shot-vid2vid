# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

def get_grid(batchsize, rows, cols, gpu_id=0):
    hor = torch.linspace(-1.0, 1.0, cols)
    hor.requires_grad = False
    hor = hor.view(1, 1, 1, cols)
    hor = hor.expand(batchsize, 1, rows, cols)
    ver = torch.linspace(-1.0, 1.0, rows)
    ver.requires_grad = False
    ver = ver.view(1, 1, rows, 1)
    ver = ver.expand(batchsize, 1, rows, cols)

    t_grid = torch.cat([hor, ver], 1)
    t_grid.requires_grad = False

    return t_grid.cuda(gpu_id)

def concat(a, b, dim=0):
    if isinstance(a, list):
        return [concat(ai, bi, dim) for ai, bi in zip(a, b)]
    if a is None:
        return b
    return torch.cat([a, b], dim=dim)

def batch_conv(x, weight, bias=None, stride=1, group_size=-1):    
    if weight is None: return x
    if isinstance(weight, list) or isinstance(weight, tuple):
        weight, bias = weight
    padding = weight.size()[-1] // 2
    groups = group_size//weight.size()[2] if group_size != -1 else 1    
    if bias is None: bias = [None] * x.size()[0]
    y = None
    for i in range(x.size()[0]):            
        if stride >= 1:
            yi = F.conv2d(x[i:i+1], weight=weight[i], bias=bias[i], padding=padding, stride=stride, groups=groups)
        else:
            yi = F.conv_transpose2d(x[i:i+1], weight=weight[i], bias=bias[i,:weight.size(2)], padding=padding, stride=int(1/stride),
                output_padding=padding, groups=groups)
        y = concat(y, yi)
    return y

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(self)
        print('Total number of parameters: %d' % num_params)

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)                
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)  

    def load_pretrained_net(self, net_src, net_dst):
        source_weights = net_src.state_dict()
        target_weights = net_dst.state_dict()
        
        for k, v in source_weights.items():             
            if k in target_weights and target_weights[k].size() == v.size():                
                target_weights[k] = v
        net_dst.load_state_dict(target_weights)

    def resample(self, image, flow):        
        b, c, h, w = image.size()
        grid = get_grid(b, h, w, gpu_id=flow.get_device())                
        flow = torch.cat([flow[:, 0:1, :, :] / ((w - 1.0) / 2.0), flow[:, 1:2, :, :] / ((h - 1.0) / 2.0)], dim=1)        
        final_grid = (grid + flow).permute(0, 2, 3, 1).cuda(image.get_device())
        output = torch.nn.functional.grid_sample(image, final_grid, mode='bilinear', padding_mode='border')
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return z   

    def sum(self, x):
        if type(x) != list: return x
        return sum([self.sum(xi) for xi in x])

    def sum_mul(self, x):
        assert(type(x) == list)
        if type(x[0]) != list:
            return np.prod(x) + x[0]
        return [self.sum_mul(xi) for xi in x]

    def split_weights(self, weight, sizes):
        if isinstance(sizes, list):
            weights = []
            cur_size = 0                        
            for i in range(len(sizes)):                
                next_size = cur_size + self.sum(sizes[i])
                weights.append(self.split_weights(weight[:,cur_size:next_size], sizes[i]))
                cur_size = next_size
            assert(next_size == weight.size()[1])
            return weights
        return weight

    def reshape_weight(self, x, weight_size):        
        if type(weight_size[0]) == list and type(x) != list:
            x = self.split_weights(x, self.sum_mul(weight_size))        
        if type(x) == list:
            return [self.reshape_weight(xi, wi) for xi, wi in zip(x, weight_size)]
        weight_size = [x.size()[0]] + weight_size
        bias_size = weight_size[1]
        try:
            weight = x[:, :-bias_size].view(weight_size)
            bias = x[:, -bias_size:]
        except:
            weight = x.view(weight_size)
            bias = None
        return [weight, bias]

    def reshape_embed_input(self, x):
        if isinstance(x, list):
            return [self.reshape_embed_input(xi) for xi in zip(x)]            
        b, c, _, _ = x.size()
        x = x.view(b*c, -1)        
        return x
