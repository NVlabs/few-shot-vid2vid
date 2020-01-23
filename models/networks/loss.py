# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.networks.vgg import VGG_Activations, Vgg19

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
            return self.fake_label_tensor.expand_as(input)

    def loss(self, input, target_is_real, weight=None, reduce_dim=True, for_discriminator=True):
        if self.gan_mode == 'original':
            target_tensor = self.get_target_tensor(input, target_is_real)
            batchsize = input.size(0)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor, weight=weight)
            if not reduce_dim:
                loss = loss.view(batchsize, -1).mean(dim=1)
            return loss
        elif self.gan_mode == 'ls':
            #target_tensor = self.get_target_tensor(input, target_is_real)
            target_tensor = input * 0 + (self.real_label if target_is_real else self.fake_label)
            if weight is None and reduce_dim:
                return F.mse_loss(input, target_tensor)
            error = (input - target_tensor)**2
            if weight is not None:
                error *= weight
            if reduce_dim:
                return torch.mean(error)
            else:
                return error.view(input.size(0), -1).mean(dim=1)
        elif self.gan_mode == 'hinge':
            assert weight == None
            assert reduce_dim == True
            if for_discriminator:                
                if target_is_real:
                    minval = torch.min(input - 1, input * 0)
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, input * 0)
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"                
                loss = -torch.mean(input)

            return loss
        else:
            # wgan
            assert weight is None and reduce_dim
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, weight=None, reduce_dim=True, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, weight, reduce_dim, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, weight, reduce_dim, for_discriminator)

        
class VGGLoss(nn.Module):
    def __init__(self, opt, gpu_ids):
        super(VGGLoss, self).__init__()                   
        self.vgg = VGG_Activations([1, 6, 11, 20, 29]).cuda()        
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def compute_loss(self, x_vgg, y_vgg):
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

    def forward(self, x, y):
        if len(x.size()) == 5:
            b, t, c, h, w = x.size()
            x, y = x.view(-1, c, h, w), y.view(-1, c, h, w)
        
        y_vgg = self.vgg(y)        
        x_vgg = self.vgg(x)
        loss = self.compute_loss(x_vgg, y_vgg)
        return loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):        
        mask = mask.expand_as(input)
        loss = self.criterion(input * mask, target * mask)
        return loss

class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    



