# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import numpy as np

from models.networks.sync_batchnorm import DataParallelWithCallback
from models.vid2vid_model import Vid2VidModel
from util.distributed import master_only
from util.distributed import master_only_print as print

def create_model(opt, epoch=0):            
    model = Vid2VidModel()
    model.initialize(opt, epoch)
    print("model [%s] was created" % (model.name()))
    
    if opt.isTrain:        
        model = WrapModel(opt, model)
        flowNet = None
        if not opt.no_flow_gt:
            from .flownet import FlowNet
            flowNet = FlowNet()
            flowNet.initialize(opt)
            flowNet = WrapModel(opt, flowNet)
        return model, flowNet
    return model

def WrapModel(opt, model):
    if opt.distributed:
        model = model.cuda(opt.gpu_ids[0])
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model = MyModel(opt, model)
    return model

@master_only
def save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, model, end_of_epoch=False):
    if not end_of_epoch:
        if (total_steps % opt.save_latest_freq == 0):
            visualizer.vis_print(opt, 'saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save_networks('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            model.cuda()
    else:
        if epoch % opt.save_epoch_freq == 0:
            visualizer.vis_print(opt, 'saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
            model.module.save_networks('latest')            
            model.module.save_networks(epoch)            
            np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')
            model.cuda()

def update_models(opt, epoch, model, data_loader):
    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate(epoch)

    ### train single frame first then sequence of frames
    if epoch == opt.niter_single + 1 and not model.module.temporal:
        model.module.make_temporal_model()

    ### gradually grow training sequence length
    epoch_temp = epoch - opt.niter_single
    if epoch_temp > 0 and ((epoch_temp - 1) % opt.niter_step) == 0:
        data_loader.dataset.update_training_batch((epoch_temp - 1) // opt.niter_step)

class MyModel(nn.Module):
    def __init__(self, opt, model):        
        super(MyModel, self).__init__()
        self.opt = opt
        model = model.cuda(opt.gpu_ids[0])
        self.module = model
        
        if opt.distributed:            
            self.model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        else:
            #self.model = nn.DataParallel(model, device_ids=opt.gpu_ids)    
            self.model = DataParallelWithCallback(model, device_ids=opt.gpu_ids)
        if opt.batch_for_first_gpu != -1:
            self.bs_per_gpu = (opt.batchSize - opt.batch_for_first_gpu) // (len(opt.gpu_ids) - 1) # batch size for each GPU
        else:
            self.bs_per_gpu = int(np.ceil(float(opt.batchSize) / len(opt.gpu_ids))) # batch size for each GPU
        self.pad_bs = self.bs_per_gpu * len(opt.gpu_ids) - opt.batchSize           

    def forward(self, *inputs, **kwargs):
        inputs = self.add_dummy_to_tensor(inputs, self.pad_bs)
        outputs = self.model(*inputs, **kwargs, dummy_bs=self.pad_bs)
        if self.pad_bs == self.bs_per_gpu: # gpu 0 does 0 batch but still returns 1 batch
            return self.remove_dummy_from_tensor(outputs, 1)
        return outputs        

    def add_dummy_to_tensor(self, tensors, add_size=0):        
        if add_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
                
        if isinstance(tensors, torch.Tensor):            
            dummy = torch.zeros_like(tensors)[:add_size]
            tensors = torch.cat([dummy, tensors])
        return tensors

    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if remove_size == 0 or tensors is None: return tensors
        if type(tensors) == list or type(tensors) == tuple:
            return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
        
        if isinstance(tensors, torch.Tensor):
            tensors = tensors[remove_size:]
        return tensors