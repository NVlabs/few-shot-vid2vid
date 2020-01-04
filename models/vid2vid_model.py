# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import numpy as np

import models.networks as networks
from models.base_model import BaseModel
from models.input_process import *
from models.loss_collector import LossCollector, loss_backward
from models.face_refiner import FaceRefineModel

class Vid2VidModel(BaseModel):
    def name(self):
        return 'Vid2VidModel'

    def initialize(self, opt, epoch=0):
        BaseModel.initialize(self, opt)        
        torch.backends.cudnn.benchmark = True        
        
        # define losses        
        self.lossCollector = LossCollector()
        self.lossCollector.initialize(opt)        
        
        # Face network        
        self.refine_face = hasattr(opt, 'refine_face') and opt.refine_face
        self.faceRefiner = None
        if self.refine_face or self.add_face_D:
            self.faceRefiner = FaceRefineModel()
            self.faceRefiner.initialize(opt, self.add_face_D, self.refine_face)   

        # define networks
        self.define_networks(epoch)

        # load networks        
        self.load_networks()        

    def forward(self, data_list, save_images=False, mode='inference', dummy_bs=0):              
        tgt_label, tgt_image, flow_gt, conf_gt, ref_labels, ref_images, \
            prev_label, prev_image = encode_input(self.opt, data_list, dummy_bs)
        
        if mode == 'generator':            
            g_loss, generated, prev = self.forward_generator(tgt_label, tgt_image, ref_labels, ref_images, 
                prev_label, prev_image, flow_gt, conf_gt)
            return g_loss, generated if save_images else [], prev

        elif mode == 'discriminator':            
            d_loss = self.forward_discriminator(tgt_label, tgt_image, ref_labels, ref_images, prev_label, prev_image)
            return d_loss    

        else:
            return self.inference(tgt_label, ref_labels, ref_images)
   
    def forward_generator(self, tgt_label, tgt_image, ref_labels, ref_images, prev_label=None, prev_image=None, 
                          flow_gt=[None]*2, conf_gt=[None]*2):
        opt = self.opt        
        ### fake generation
        [fake_image, fake_raw_image, warped_image, flow, weight], [fg_mask, ref_fg_mask], \
            [ref_label, ref_image], [prev_label, prev_image], atn_score = \
            self.generate_images(tgt_label, ref_labels, ref_images, [prev_label, prev_image])

        ### temporal losses
        nets = self.netD, self.netDT, self.netDf, self.faceRefiner
        loss_GT_GAN, loss_GT_GAN_Feat = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        if self.isTrain:                        
            tgt_image_all = torch.cat([prev_image, tgt_image], dim=1)
            fake_image_all = torch.cat([prev_image, fake_image], dim=1)
            data_list = [None, tgt_image_all, fake_image_all, None, None]
            loss_GT_GAN, loss_GT_GAN_Feat = self.lossCollector.compute_GAN_losses(nets, data_list, 
                for_discriminator=False, for_temporal=True)

        ### individual frame losses
        # GAN loss
        fg_mask_union = combine_fg_mask(fg_mask, ref_fg_mask, self.has_fg)        
        data_list = [tgt_label, [tgt_image, tgt_image * fg_mask_union], [fake_image, fake_raw_image], ref_label, ref_image]
        loss_G_GAN, loss_G_GAN_Feat, loss_Gf_GAN, loss_Gf_GAN_Feat = self.lossCollector.compute_GAN_losses(nets,
            data_list, for_discriminator=False)
                   
        # VGG loss        
        loss_G_VGG = self.lossCollector.compute_VGG_losses(fake_image, fake_raw_image, tgt_image, fg_mask_union)        

        # flow loss
        flow, weight, flow_gt, conf_gt, fg_mask, ref_fg_mask, warped_image, tgt_image = \
            self.reshape([flow, weight, flow_gt, conf_gt, fg_mask, ref_fg_mask, warped_image, tgt_image])             
        loss_F_Flow, loss_F_Warp, body_mask_diff = self.lossCollector.compute_flow_losses(flow, warped_image, tgt_image, 
            flow_gt, conf_gt, fg_mask, tgt_label, ref_label, self.netG)
        loss_W = self.lossCollector.compute_weight_losses(weight, fake_image, warped_image, tgt_label, tgt_image, 
            fg_mask, ref_fg_mask, body_mask_diff)
        
        loss_list = [loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, # GAN + VGG loss
                     loss_Gf_GAN, loss_Gf_GAN_Feat,           # additional GAN loss for face
                     loss_GT_GAN, loss_GT_GAN_Feat,           # temporal GAN loss
                     loss_F_Flow, loss_F_Warp, loss_W]        # flow loss
        loss_list = [loss.view(1, 1) for loss in loss_list]
            
        return loss_list, \
               [fake_image, fake_raw_image, warped_image, flow, weight, atn_score], \
               [prev_label, prev_image]   
    
    def forward_discriminator(self, tgt_label, tgt_image, ref_labels, ref_images, prev_label=None, prev_image=None):
        ### Fake Generation
        with torch.no_grad():
            [fake_image, fake_raw_image, _, _, _], [fg_mask, ref_fg_mask], [ref_label, ref_image], _, _ = \
                self.generate_images(tgt_label, ref_labels, ref_images, [prev_label, prev_image])

        ### temporal losses
        nets = self.netD, self.netDT, self.netDf, self.faceRefiner
        loss_temp = []
        if self.isTrain:
            if prev_image is None: prev_image = tgt_image.repeat(1, self.opt.n_frames_G-1, 1, 1, 1)
            tgt_image_all = torch.cat([prev_image, tgt_image], dim=1)
            fake_image_all = torch.cat([prev_image, fake_image], dim=1)            
            data_list = [None, tgt_image_all, fake_image_all, None, None]
            loss_temp = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=True, for_temporal=True)

        ### individual frame losses        
        fg_mask_union = combine_fg_mask(fg_mask, ref_fg_mask, self.has_fg)        
        data_list = [tgt_label, [tgt_image, tgt_image * fg_mask_union], [fake_image, fake_raw_image], ref_label, ref_image]        
        loss_indv = self.lossCollector.compute_GAN_losses(nets, data_list, for_discriminator=True)

        loss_list = list(loss_indv) + list(loss_temp)
        loss_list = [loss.view(1, 1) for loss in loss_list]
        return loss_list              

    def generate_images(self, tgt_labels, ref_labels, ref_images, prevs=[None, None]):
        opt = self.opt      
        generated_images, atn_score = [None] * 5, None 
        generated_masks = [None] * 2 if self.has_fg else [1, 1]
        ref_labels_valid = use_valid_labels(opt, ref_labels)
        
        for t in range(opt.n_frames_per_gpu):
            # get inputs for time t            
            tgt_label_t, tgt_label_valid, prev_t = self.get_input_t(tgt_labels, prevs, t)
                                  
            # actual network forward
            fake_image, flow, weight, fake_raw_image, warped_image, mu, logvar, atn_score, ref_idx \
                = self.netG(tgt_label_valid, ref_labels_valid, ref_images, prev_t)
            
            ref_label_valid, ref_label_t, ref_image_t = self.netG.pick_ref([ref_labels_valid, ref_labels, ref_images], ref_idx)
            # refine face if necessary            
            if self.refine_face:                                
                fake_image = self.faceRefiner.refine_face_region(self.netGf, tgt_label_valid, fake_image, tgt_label_t,
                    ref_label_valid, ref_image_t, ref_label_t)
                        
            # concatenate current output with previous outputs
            fg_mask, ref_fg_mask = get_fg_mask(self.opt, [tgt_label_t, ref_label_t], self.has_fg)
            if fake_raw_image is not None:
                fake_raw_image = fake_raw_image * combine_fg_mask(fg_mask, ref_fg_mask, self.has_fg)
            generated_images = self.concat([generated_images, [fake_image, fake_raw_image, warped_image, flow, weight]], dim=1)
            generated_masks = self.concat([generated_masks, [fg_mask, ref_fg_mask]], dim=1)            
            prevs = self.concat_prev(prevs, [tgt_label_valid, fake_image])
                
        return generated_images, generated_masks, [ref_label_valid, ref_image_t], prevs, atn_score

    def get_input_t(self, tgt_labels, prevs, t):
        b, _, _, h, w = tgt_labels.shape        
        tgt_label = tgt_labels[:,t]
        tgt_label_valid = use_valid_labels(self.opt, tgt_label)
        prevs = [prev.contiguous().view(b, -1, h, w) if prev is not None else None for prev in prevs]        
        return tgt_label, tgt_label_valid, prevs

    def concat_prev(self, prev, now):
        if type(prev) == list:
            return [self.concat_prev(p, n) for p, n in zip(prev, now)]
        if prev is None:
            prev = now.unsqueeze(1).repeat(1, self.opt.n_frames_G-1, 1, 1, 1)
        else:
            prev = torch.cat([prev[:, 1:], now.unsqueeze(1)], dim=1)
        return prev.detach()
    
    ########################################### inference ###########################################
    def inference(self, tgt_label, ref_labels, ref_images):
        opt = self.opt
        if not hasattr(self, 'prevs') or self.prevs is None:
            print('first image')            
            self.prevs = prevs = [None, None]
            self.t = 0
        else:            
            b, _, _, h, w = tgt_label.shape
            prevs = [prev.view(b, -1, h, w) for prev in self.prevs]            
            self.t += 1        
                        
        tgt_label_valid, ref_labels_valid = use_valid_labels(opt, [tgt_label[:,-1], ref_labels])
        if opt.finetune and self.t == 0:
            self.finetune(ref_labels, ref_images)

        with torch.no_grad():            
            fake_image, flow, weight, fake_raw_image, warped_image, _, _, atn_score, ref_idx = self.netG(tgt_label_valid, 
                ref_labels_valid, ref_images, prevs, t=self.t)

            ref_label_valid, ref_label, ref_image = self.netG.pick_ref([ref_labels_valid, ref_labels, ref_images], ref_idx)
            if self.refine_face:                                
                fake_image = self.faceRefiner.refine_face_region(self.netGf, tgt_label_valid, fake_image, tgt_label[:,-1], 
                    ref_label_valid, ref_image, ref_label)            
            
            self.prevs = self.concat_prev(self.prevs, [tgt_label_valid, fake_image])            
            
        return fake_image, fake_raw_image, warped_image, flow, weight, atn_score

    def finetune(self, ref_labels, ref_images):
        train_names = ['fc', 'conv_img', 'up']        
        params, _ = self.get_train_params(self.netG, train_names) 
        if self.refine_face: params += list(self.netGf.parameters())
        self.optimizer_G = self.get_optimizer(params, for_discriminator=False)
        
        update_D = True
        if update_D:
            params = list(self.netD.parameters())
            if self.add_face_D: params += list(self.netDf.parameters())            
            self.optimizer_D = self.get_optimizer(params, for_discriminator=True)

        iterations = 100
        for it in range(1, iterations + 1):            
            idx = np.random.randint(ref_labels.size(1))
            tgt_label, tgt_image = ref_labels[:,idx].unsqueeze(1), ref_images[:,idx].unsqueeze(1)            

            g_losses, generated, prev = self.forward_generator(tgt_label, tgt_image, ref_labels, ref_images)
            g_losses = loss_backward(self.opt, g_losses, self.optimizer_G)

            d_losses = []
            if update_D:
                d_losses = self.forward_discriminator(tgt_label, tgt_image, ref_labels, ref_images)
                d_losses = loss_backward(self.opt, d_losses, self.optimizer_D)

            if (it % 10) == 0: 
                message = '(iters: %d) ' % it
                loss_dict = dict(zip(self.lossCollector.loss_names, g_losses + d_losses))
                for k, v in loss_dict.items():
                    if v != 0: message += '%s: %.3f ' % (k, v)
                print(message)