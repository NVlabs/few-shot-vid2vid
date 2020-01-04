# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import copy
import torch
from util.visualizer import Visualizer
import models.networks as networks
from util.distributed import master_only_print as print

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)        
        self.old_lr = opt.lr

        self.pose = 'pose' in opt.dataset_mode        
        self.face = 'face' in opt.dataset_mode        
        self.street = 'street' in opt.dataset_mode
        self.warp_ref = opt.warp_ref
        self.has_fg = self.pose    
        self.add_face_D = opt.add_face_D
        self.concat_ref_for_D = (opt.isTrain or opt.finetune) and opt.netD_subarch == 'n_layers'
        self.concat_fg_mask_for_D = self.has_fg                

    def forward(self):
        pass

    def get_optimizer(self, params, for_discriminator=False):
        opt = self.opt
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, 0.999
            G_lr, D_lr = opt.lr, opt.lr
        else:                
            beta1, beta2 = 0, opt.beta2
            G_lr, D_lr = opt.lr/2, opt.lr*2
        lr = D_lr if for_discriminator else G_lr           
        return torch.optim.Adam(params, lr=lr, betas=(beta1, beta2))        

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir

        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            Visualizer.vis_print(self.opt, '%s not exists yet!' % save_path)            
        else:               
            try:
                loaded_weights = torch.load(save_path)
                network.load_state_dict(loaded_weights)

                Visualizer.vis_print(self.opt, 'network loaded from %s' % save_path)
            except:                
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    Visualizer.vis_print(self.opt, 'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    Visualizer.vis_print(self.opt, 'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)                    
                    not_initialized = set()                    
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add('.'.join(k.split('.')[:2]))
                            if 'flow_network_temp' in k:
                                network.flow_temp_is_initalized = False
                    Visualizer.vis_print(self.opt, sorted(not_initialized))
                    network.load_state_dict(model_dict)


    def remove_dummy_from_tensor(self, tensors, remove_size=0):
        if self.isTrain and tensors[0].get_device() == 0:
            if remove_size == 0: return tensors
            if isinstance(tensors, list):
                return [self.remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]
            if tensors is None: return None
            if isinstance(tensors, torch.Tensor):
                tensors = tensors[remove_size:]
        return tensors

    def concat(self, tensors, dim=0):        
        if tensors[0] is not None and tensors[1] is not None:
            if isinstance(tensors[0], list):                
                tensors_cat = []
                for i in range(len(tensors[0])):                    
                    tensors_cat.append(self.concat([tensors[0][i], tensors[1][i]], dim=dim))                
                return tensors_cat            
            return torch.cat([tensors[0], tensors[1].unsqueeze(1)], dim=dim)
        elif tensors[1] is not None:            
            if isinstance(tensors[1], list):
                return [t.unsqueeze(1) if t is not None else None for t in tensors[1]]
            return tensors[1].unsqueeze(1)
        return tensors[0]

    def reshape(self, tensors, for_temporal=False):
        if isinstance(tensors, list):
            return [self.reshape(tensor, for_temporal) for tensor in tensors]        
        if tensors is None or (type(tensors) != torch.Tensor) or len(tensors.size()) <= 4: return tensors
        bs, t, ch, h, w = tensors.size()
        if not for_temporal:
            tensors = tensors.contiguous().view(-1, ch, h, w)
        elif not self.opt.isTrain:
            tensors = tensors.contiguous().view(bs, -1, h, w)
        else:
            nD = self.tD
            if t > nD:
                if t % nD == 0:
                    tensors = tensors.contiguous().view(-1, ch*nD, h, w)
                else:
                    n = t // nD
                    tensors = tensors[:, -n*nD:].contiguous().view(-1, ch*nD, h, w)
            else:
                tensors = tensors.contiguous().view(bs, ch*t, h, w)
        return tensors    

    def divide_pred(self, pred):        
        if type(pred) == list:            
            fake = [[tensor[:tensor.size(0)//2] for tensor in p] for p in pred]
            real = [[tensor[tensor.size(0)//2:] for tensor in p] for p in pred]
            return fake, real
        else:
            return torch.chunk(pred, 2, dim=0)                    

    def get_train_params(self, netG, train_names):
        train_list = set()
        params = []          
        params_dict = netG.state_dict()                
        for key, value in params_dict.items():
            do_train = False
            for model_name in train_names:
                if model_name in key: do_train = True            
            if do_train:
                module = netG                        
                key_list = key.split('.')
                for k in key_list:
                    module = getattr(module, k)
                params += [module]
                train_list.add('.'.join(key_list[:1]))
        print('training layers: ', train_list)
        return params, train_list

    def define_networks(self, start_epoch):
        opt = self.opt        
        # Generator network        
        input_nc = opt.label_nc if (opt.label_nc != 0 and not self.pose) else opt.input_nc
        netG_input_nc = input_nc           
        opt.for_face = False        
        self.netG = networks.define_G(opt)        
        if self.refine_face:            
            opt_face = copy.deepcopy(opt)
            opt_face.n_downsample_G -= 1
            if opt_face.n_adaptive_layers > 0: opt_face.n_adaptive_layers -= 1
            opt_face.input_nc = opt.output_nc
            opt_face.fineSize = self.faceRefiner.face_size
            opt_face.aspect_ratio = 1
            opt_face.for_face = True
            self.netGf = networks.define_G(opt_face)

        # Discriminator network
        if self.isTrain or opt.finetune:            
            netD_input_nc = input_nc + opt.output_nc + (1 if self.concat_fg_mask_for_D else 0)
            if self.concat_ref_for_D:
                netD_input_nc *= 2
            self.netD = networks.define_D(opt, netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm_D, opt.netD_subarch, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)            
            if self.add_face_D:
                self.netDf = networks.define_D(opt, opt.output_nc * 2, opt.ndf, opt.n_layers_D, opt.norm_D, 'n_layers',
                                               1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            else:
                self.netDf = None
        self.temporal = False
        self.netDT = None             
                    
        print('---------- Networks initialized -------------')

        # initialize optimizers
        if self.isTrain:            
            # optimizer G
            params = list(self.netG.parameters())           
            if self.refine_face: params += list(self.netGf.parameters())
            self.optimizer_G = self.get_optimizer(params, for_discriminator=False)

            # optimizer D            
            params = list(self.netD.parameters())
            if self.add_face_D: params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)           

        print('---------- Optimizers initialized -------------')

        # make model temporal by generating multiple frames
        if (not opt.isTrain or start_epoch > opt.niter_single) and opt.n_frames_G > 1:
            self.make_temporal_model() 

    def save_networks(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)        
        if self.refine_face:
            self.save_network(self.netGf, 'Gf', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)        
        if self.temporal:
            self.save_network(self.netDT, 'DT', which_epoch, self.gpu_ids)        
        if self.add_face_D:
            self.save_network(self.netDf, 'Df', which_epoch, self.gpu_ids)                

    def load_networks(self):
        opt = self.opt
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain or opt.continue_train else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)                  
            if self.temporal and opt.warp_ref and not self.netG.flow_temp_is_initalized:
                self.netG.load_pretrained_net(self.netG.flow_network_ref, self.netG.flow_network_temp)
            if self.refine_face:
                self.load_network(self.netGf, 'Gf', opt.which_epoch, pretrained_path)  
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
                if self.temporal: 
                    self.load_network(self.netDT, 'DT', opt.which_epoch, pretrained_path)
                if self.add_face_D: 
                    self.load_network(self.netDf, 'Df', opt.which_epoch, pretrained_path) 

    def update_learning_rate(self, epoch):
        new_lr = self.opt.lr * (1 - (epoch - self.opt.niter) / (self.opt.niter_decay + 1))
        if self.opt.no_TTUR:            
            G_lr, D_lr = new_lr, new_lr
        else:                            
            G_lr, D_lr = new_lr/2, new_lr*2
        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = D_lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = G_lr
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

    def make_temporal_model(self):
        opt = self.opt
        self.temporal = True
        self.netG.set_flow_prev()
        self.netG.cuda()

        if opt.isTrain:
            self.lossCollector.tD = min(opt.n_frames_D, opt.n_frames_G)  
            if opt.finetune_all:      
                params = list(self.netG.parameters())
            else:
                train_names = ['flow_network_temp']
                if opt.spade_combine: 
                    train_names += ['img_warp_embedding', 'mlp_gamma3', 'mlp_beta3']
                params, _ = self.get_train_params(self.netG, train_names) 
                    
            if self.refine_face: params += list(self.netGf.parameters())
            self.optimizer_G = self.get_optimizer(params, for_discriminator=False)
            
            # temporal discriminator
            self.netDT = networks.define_D(opt, opt.output_nc * self.lossCollector.tD, opt.ndf, opt.n_layers_D, opt.norm_D, 'n_layers',
                                           1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            # optimizer D            
            params = list(self.netD.parameters()) + list(self.netDT.parameters())
            if self.add_face_D: params += list(self.netDf.parameters())
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)           

            print('---------- Now start training multiple frames -------------')
