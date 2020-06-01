# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import numpy as np
from util.image_pool import ImagePool
from models.base_model import BaseModel
import models.networks as networks
from models.input_process import *
from models.networks.base_network import resample

class LossCollector(BaseModel):
    def name(self):
        return 'LossCollector'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)        
        
        # define losses
        self.define_losses() 
        self.tD = 1           

    ####################################### loss related ################################################
    def define_losses(self):
        opt = self.opt
        # set loss functions
        if self.isTrain or opt.finetune:
            self.fake_pool = ImagePool(0)
            self.old_lr = opt.lr
                
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.Tensor, opt=opt)            
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionFlow = networks.MaskedL1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(opt, self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names_G = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'Gf_GAN', 'Gf_GAN_feat', 'GT_GAN', 'GT_GAN_Feat', 
                                 'F_Flow', 'F_Warp', 'F_Mask']
            self.loss_names_D = ['D_real', 'D_fake', 'Df_real', 'Df_fake', 'DT_real', 'DT_fake'] 
            self.loss_names = self.loss_names_G + self.loss_names_D

    def discriminate(self, netD, tgt_label, fake_image, tgt_image, ref_image, for_discriminator):                
        tgt_concat = torch.cat([fake_image, tgt_image], dim=0)        
        if tgt_label is not None:
            tgt_concat = torch.cat([tgt_label.repeat(2, 1, 1, 1), tgt_concat], dim=1)
            
        if ref_image is not None:             
            ref_image = ref_image.repeat(2, 1, 1, 1)
            if self.concat_ref_for_D:                
                tgt_concat = torch.cat([ref_image, tgt_concat], dim=1)
                ref_image = None        

        discriminator_out = netD(tgt_concat, ref_image)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        if for_discriminator:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)            
            return [loss_D_real, loss_D_fake]
        else:
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_GAN_Feat = self.GAN_matching_loss(pred_real, pred_fake, for_discriminator)            
            return [loss_G_GAN, loss_G_GAN_Feat]

    def discriminate_face(self, netDf, fake_image, tgt_label, tgt_image, ref_label, ref_image, faceRefiner, for_discriminator):
        losses = [self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)]
        if self.add_face_D:            
            real_region, fake_region = faceRefiner.crop_face_region([tgt_image, fake_image], tgt_label)
            ref_region = faceRefiner.crop_face_region(ref_image, ref_label)
            
            losses = self.discriminate(netDf, ref_region, fake_region, real_region, None, for_discriminator=for_discriminator)
            losses = [loss * self.opt.lambda_face for loss in losses]
            if for_discriminator: 
                return losses
            else: 
                loss_Gf_GAN, loss_Gf_GAN_Feat = losses                
                loss_Gf_GAN_Feat += self.criterionFeat(fake_region, real_region) * self.opt.lambda_feat
                loss_Gf_GAN_Feat += self.criterionVGG(fake_region, real_region) * self.opt.lambda_vgg
            return [loss_Gf_GAN, loss_Gf_GAN_Feat]
        return losses

    def compute_GAN_losses(self, nets, data_list, for_discriminator, for_temporal=False):        
        if for_temporal and self.tD < 2:
            return [self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)]
        tgt_label, tgt_image, fake_image, ref_label, ref_image = data_list
        netD, netDT, netDf, faceRefiner = nets
        if isinstance(fake_image, list):
            fake_image = [x for x in fake_image if x is not None]
            losses = [self.compute_GAN_losses(nets, [tgt_label, real_i, fake_i, ref_label, ref_image], 
                for_discriminator, for_temporal) for fake_i, real_i in zip(fake_image, tgt_image)]
            return [sum([item[i] for item in losses]) for i in range(len(losses[0]))]
                        
        tgt_label, tgt_image, fake_image = self.reshape([tgt_label, tgt_image, fake_image], for_temporal)
        
        # main discriminator loss        
        input_label = ref_concat = None
        if not for_temporal:            
            t = self.opt.n_frames_per_gpu
            ref_label, ref_image = ref_label.repeat(t,1,1,1), ref_image.repeat(t,1,1,1)                  
            input_label = use_valid_labels(self.opt, tgt_label)                
            if self.concat_fg_mask_for_D:
                fg_mask, ref_fg_mask = get_fg_mask(self.opt, [tgt_label, ref_label], self.has_fg)
                input_label = torch.cat([input_label, fg_mask], dim=1)
                ref_label = torch.cat([ref_label, ref_fg_mask], dim=1)                        
            ref_concat = torch.cat([ref_label, ref_image], dim=1)

        netD = netD if not for_temporal else netDT        
        losses = self.discriminate(netD, input_label, fake_image, tgt_image, ref_concat, for_discriminator=for_discriminator)
        if for_temporal: 
            if not for_discriminator: losses = [loss * self.opt.lambda_temp for loss in losses]
            return losses

        # additional GAN loss (for face region)
        losses_face = self.discriminate_face(netDf, fake_image, tgt_label, tgt_image, ref_label, ref_image, faceRefiner, for_discriminator)        
        return losses + losses_face 

    def compute_VGG_losses(self, fake_image, fake_raw_image, tgt_image, fg_mask_union):
        loss_G_VGG = self.Tensor(1).fill_(0)
        opt = self.opt
        if not opt.no_vgg_loss:
            if fake_image is not None:
                loss_G_VGG = self.criterionVGG(fake_image, tgt_image)
            if fake_raw_image is not None:
                loss_G_VGG += self.criterionVGG(fake_raw_image, tgt_image * fg_mask_union)
        return loss_G_VGG * opt.lambda_vgg

    def compute_flow_losses(self, flow, warped_image, tgt_image, flow_gt, flow_conf_gt, fg_mask, tgt_label, ref_label):
        loss_F_Flow_r, loss_F_Warp_r = self.compute_flow_loss(flow[0], warped_image[0], tgt_image, flow_gt[0], flow_conf_gt[0], fg_mask)
        loss_F_Flow_p, loss_F_Warp_p = self.compute_flow_loss(flow[1], warped_image[1], tgt_image, flow_gt[1], flow_conf_gt[1], fg_mask)
        loss_F_Flow = loss_F_Flow_p + loss_F_Flow_r
        loss_F_Warp = loss_F_Warp_p + loss_F_Warp_r
        lambda_flow = self.opt.lambda_flow
        
        body_mask_diff = None
        # Get the disocclusion regions for (DensePose) body parts and foreground.
        if self.opt.isTrain and self.pose and flow[0] is not None:            
            body_mask = get_part_mask(tgt_label[:,:,2])
            ref_body_mask = get_part_mask(ref_label[:,2].unsqueeze(1)).expand_as(body_mask)
            body_mask, ref_body_mask = self.reshape([body_mask, ref_body_mask])            
            ref_body_mask_warp = resample(ref_body_mask, flow[0])
            loss_F_Warp += self.criterionFeat(ref_body_mask_warp, body_mask)

            if self.has_fg:
                fg_mask, ref_fg_mask = get_fg_mask(self.opt, [tgt_label, ref_label], True)
                ref_fg_mask_warp = resample(ref_fg_mask, flow[0])
                loss_F_Warp += self.criterionFeat(ref_fg_mask_warp, fg_mask)

            body_mask_diff = torch.sum(abs(ref_body_mask_warp - body_mask), dim=1, keepdim=True)
        return loss_F_Flow * lambda_flow, loss_F_Warp * lambda_flow, body_mask_diff

    def compute_flow_loss(self, flow, warped_image, tgt_image, flow_gt, flow_conf_gt, fg_mask):        
        loss_F_Flow, loss_F_Warp = self.Tensor(1).fill_(0), self.Tensor(1).fill_(0)
        if self.opt.isTrain and flow is not None:
            if flow_gt is not None and self.opt.n_shot == 1: # Only computed ground truth flow for first reference image.
                loss_F_Flow = self.criterionFlow(flow, flow_gt, flow_conf_gt * fg_mask)
            loss_F_Warp = self.criterionFeat(warped_image, tgt_image)
        return loss_F_Flow, loss_F_Warp

    def compute_mask_losses(self, flow_mask, fake_image, warped_image, tgt_label, tgt_image, fake_raw_image,
            fg_mask, ref_fg_mask, body_mask_diff):        
        fake_raw_image = fake_raw_image[:,-1] if fake_raw_image is not None else None        
        loss_mask = self.Tensor(1).fill_(0)
        loss_mask += self.compute_mask_loss(flow_mask[0], warped_image[0], tgt_image, fake_image[:,-1], fake_raw_image)
        loss_mask += self.compute_mask_loss(flow_mask[1], warped_image[1], tgt_image, fake_image[:,-1], fake_raw_image)
        
        opt = self.opt
        if opt.isTrain and self.pose and self.warp_ref:
            # Force the output to use more from the warped image for face region.
            flow_mask_ref = flow_mask[0]
            b, t, _, h, w = tgt_label.size()
            dummy0, dummy1 = torch.zeros_like(flow_mask_ref), torch.ones_like(flow_mask_ref)
            face_mask = get_face_mask(tgt_label[:,:,2]).view(-1, 1, h, w)
            face_mask = torch.nn.AvgPool2d(15, padding=7, stride=1)(face_mask)                    
            loss_mask += self.criterionFlow(flow_mask_ref, dummy0, face_mask)
            if opt.spade_combine:                
                loss_mask += self.criterionFlow(fake_image[:,-1], warped_image[0].detach(), face_mask)

            # Force the output to use more from the hallucinated image for disocclusion part.
            fg_mask_diff = ((ref_fg_mask - fg_mask) > 0).float()            
            loss_mask += self.criterionFlow(flow_mask_ref, dummy1, fg_mask_diff)
            loss_mask += self.criterionFlow(flow_mask_ref, dummy1, body_mask_diff)

        return loss_mask * opt.lambda_mask

    def compute_mask_loss(self, flow_mask, warped_image, tgt_image, fake_image, fake_raw_image):
        loss_mask = 0
        if self.opt.isTrain and flow_mask is not None:
            dummy0 = torch.zeros_like(flow_mask)
            dummy1 = torch.ones_like(flow_mask)            
            
            # Compute the confidence map based on L1 distance between warped and GT image.
            img_diff = torch.sum(abs(warped_image - tgt_image), dim=1, keepdim=True)
            conf = torch.clamp(1 - img_diff, 0, 1)

            # Force mask value to be small if warped image is similar to GT, and vice versa. 
            loss_mask = self.criterionFlow(flow_mask, dummy0, conf)
            loss_mask += self.criterionFlow(flow_mask, dummy1, 1 - conf)

        return loss_mask

    def GAN_matching_loss(self, pred_real, pred_fake, for_discriminator=False):
        loss_G_GAN_Feat = self.Tensor(1).fill_(0)
        if not for_discriminator and not self.opt.no_ganFeat_loss:            
            num_D = len(pred_fake)
            D_masks = 1.0 / num_D            
            for i in range(num_D):
                for j in range(len(pred_fake[i])-1):
                    loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    loss_G_GAN_Feat += D_masks * loss
        return loss_G_GAN_Feat * self.opt.lambda_feat

def loss_backward(opt, losses, optimizer, loss_id):
    losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
    loss = sum(losses)
    optimizer.zero_grad()
    if opt.amp != 'O0':
        from apex import amp
        with amp.scale_loss(loss, optimizer, loss_id=loss_id) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    return losses