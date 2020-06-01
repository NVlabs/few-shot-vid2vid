# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

import numpy as np
import torch
import torch.nn.functional as F
import sys

from .base_model import BaseModel

class FlowNet(BaseModel):
    def name(self):
        return 'FlowNet'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # flownet 2         
        from .networks.flownet2_pytorch import models as flownet2_models
        from .networks.flownet2_pytorch.utils import tools as flownet2_tools
        from .networks.flownet2_pytorch.networks.resample2d_package.resample2d import Resample2d                

        self.flowNet = flownet2_tools.module_to_dict(flownet2_models)['FlowNet2']().cuda()
        checkpoint = torch.load('models/networks/flownet2_pytorch/FlowNet2_checkpoint.pth.tar', map_location=torch.device('cpu'))
        self.flowNet.load_state_dict(checkpoint['state_dict'])
        self.flowNet.eval() 
        self.resample = Resample2d()
        self.downsample = torch.nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)        

    def forward(self, data_list, epoch=0, dummy_bs=0):
        if data_list[0].get_device() == 0:                
            data_list = self.remove_dummy_from_tensor(data_list, dummy_bs) 
        image_now, image_ref = data_list
        image_now, image_ref = image_now[:,:,:3], image_ref[:,0:1,:3]

        flow_gt_prev = flow_gt_ref = conf_gt_prev = conf_gt_ref = None
        with torch.no_grad():                         
            if not self.opt.isTrain or epoch > self.opt.niter_single:
                image_prev = torch.cat([image_now[:,0:1], image_now[:,:-1]], dim=1)
                flow_gt_prev, conf_gt_prev = self.flowNet_forward(image_now, image_prev)

            if self.opt.warp_ref:
                flow_gt_ref, conf_gt_ref = self.flowNet_forward(image_now, image_ref.expand_as(image_now))              

            flow_gt, conf_gt = [flow_gt_ref, flow_gt_prev], [conf_gt_ref, conf_gt_prev]
            return flow_gt, conf_gt

    def flowNet_forward(self, input_A, input_B):        
        size = input_A.size()
        assert(len(size) == 4 or len(size) == 5)
        if len(size) == 5:
            b, n, c, h, w = size
            input_A = input_A.contiguous().view(-1, c, h, w)
            input_B = input_B.contiguous().view(-1, c, h, w)
            flow, conf = self.compute_flow_and_conf(input_A, input_B)
            return flow.view(b, n, 2, h, w), conf.view(b, n, 1, h, w)
        else:
            return self.compute_flow_and_conf(input_A, input_B)

    def compute_flow_and_conf(self, im1, im2):
        assert(im1.size()[1] == 3)
        assert(im1.size() == im2.size())        
        old_h, old_w = im1.size()[2], im1.size()[3]
        new_h, new_w = old_h//64*64, old_w//64*64        
        if old_h != new_h:            
            im1 = F.interpolate(im1, size=(new_h, new_w), mode='bilinear')
            im2 = F.interpolate(im2, size=(new_h, new_w), mode='bilinear')
        self.flowNet.cuda(im1.get_device())
        data1 = torch.cat([im1.unsqueeze(2), im2.unsqueeze(2)], dim=2)            
        flow1 = self.flowNet(data1)
        conf = (self.norm(im1 - self.resample(im2, flow1)) < 0.02).float()

        if old_h != new_h:
            flow1 = F.interpolate(flow1, size=(old_h, old_w), mode='bilinear') * old_h / new_h
            conf = F.interpolate(conf, size=(old_h, old_w), mode='bilinear')
        return flow1, conf
        
    def norm(self, t):
        return torch.sum(t*t, dim=1, keepdim=True)   
