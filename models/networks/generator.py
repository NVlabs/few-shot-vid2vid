# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
import copy
import pdb

from models.networks.base_network import BaseNetwork, batch_conv
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import SPADEResnetBlock, SPADEConv2d, actvn
import torch.nn.utils.spectral_norm as sn
#from models.networks.sn import spectral_norm as sn

class FewShotGenerator(BaseNetwork):    
    def __init__(self, opt):
        super().__init__()
        ########################### define params ##########################
        self.opt = opt                      
        self.n_downsample_G = n_downsample_G = opt.n_downsample_G # number of downsamplings in generator
        self.n_downsample_A = n_downsample_A = opt.n_downsample_A # number of downsamplings in attention module        
        self.nf = nf = opt.ngf                                    # base channel size
        self.nf_max = nf_max = min(1024, nf * (2**n_downsample_G))
        self.ch = ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_G + 2)]
                
        ### SPADE          
        self.norm = norm = opt.norm_G
        self.conv_ks = conv_ks = opt.conv_ks    # conv kernel size in main branch
        self.embed_ks = embed_ks = opt.embed_ks # conv kernel size in embedding network
        self.spade_ks = spade_ks = opt.spade_ks # conv kernel size in SPADE
        self.spade_combine = opt.spade_combine  # combine ref/prev frames with current using SPADE
        self.n_sc_layers = opt.n_sc_layers      # number of layers to perform spade combine        
        self.add_raw_loss = opt.add_raw_loss and opt.spade_combine
        ch_hidden = []                          # hidden channel size in SPADE module
        for i in range(n_downsample_G + 1):
            ch_hidden += [[ch[i]]] if not self.spade_combine or i >= self.n_sc_layers else [[ch[i]]*3]
        self.ch_hidden = ch_hidden


        ### adaptive SPADE / Convolution
        self.adap_spade = opt.adaptive_spade                               # use adaptive weight generation for SPADE
        self.adap_embed = opt.adaptive_spade and not opt.no_adaptive_embed # use adaptive for the label embedding network
        self.adap_conv = opt.adaptive_conv                                 # use adaptive for convolution layers in the main branch        
        self.n_adaptive_layers = opt.n_adaptive_layers if opt.n_adaptive_layers != -1 else n_downsample_G  # number of adaptive layers

        # for reference image encoding
        self.concat_label_ref = 'concat' in opt.use_label_ref # how to utilize the reference label map: concat | mul
        self.mul_label_ref = 'mul' in opt.use_label_ref        
        self.sh_fix = self.sw_fix = 32                      # output spatial size for adaptive pooling layer
        self.sw = opt.fineSize // (2**opt.n_downsample_G)   # output spatial size at the bottle neck of generator
        self.sh = int(self.sw / opt.aspect_ratio)                
     
        # weight generation
        self.n_fc_layers = n_fc_layers = opt.n_fc_layers    # number of fc layers in weight generation            

        ########################### define network ##########################
        norm_ref = norm.replace('spade', '')
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc
        ref_nc = opt.output_nc + (0 if not self.concat_label_ref else input_nc)
        self.ref_img_first = SPADEConv2d(ref_nc, nf, norm=norm_ref)
        if self.mul_label_ref: self.ref_label_first = SPADEConv2d(input_nc, nf, norm=norm_ref)        
        ref_conv = SPADEConv2d if not opt.res_for_ref else SPADEResnetBlock        

        ### reference image encoding
        for i in range(n_downsample_G):
            ch_in, ch_out = ch[i], ch[i+1]            
            setattr(self, 'ref_img_down_%d' % i, ref_conv(ch_in, ch_out, stride=2, norm=norm_ref))                    
            setattr(self, 'ref_img_up_%d' % i, ref_conv(ch_out, ch_in, norm=norm_ref))
            if self.mul_label_ref:
                setattr(self, 'ref_label_down_%d' % i, ref_conv(ch_in, ch_out, stride=2, norm=norm_ref))
                setattr(self, 'ref_label_up_%d' % i, ref_conv(ch_out, ch_in, norm=norm_ref))        
        
        ### SPADE / main branch weight generation
        if self.adap_spade or self.adap_conv:
            for i in range(self.n_adaptive_layers):
                ch_in, ch_out = ch[i], ch[i+1]
                conv_ks2 = conv_ks**2
                embed_ks2 = embed_ks**2
                spade_ks2 = spade_ks**2
                ch_h = ch_hidden[i][0]

                fc_names, fc_outs = [], []
                if self.adap_spade:                    
                    fc0_out = fcs_out = (ch_h * spade_ks2 + 1) * 2
                    fc1_out = (ch_h * spade_ks2 + 1) * (1 if ch_in != ch_out else 2)
                    fc_names += ['fc_spade_0', 'fc_spade_1', 'fc_spade_s']
                    fc_outs += [fc0_out, fc1_out, fcs_out]
                    if self.adap_embed:                        
                        fc_names += ['fc_spade_e']
                        fc_outs += [ch_in * embed_ks2 + 1]
                if self.adap_conv:
                    fc0_out = ch_out * conv_ks2 + 1
                    fc1_out = ch_in * conv_ks2 + 1
                    fcs_out = ch_out + 1
                    fc_names += ['fc_conv_0', 'fc_conv_1', 'fc_conv_s']
                    fc_outs += [fc0_out, fc1_out, fcs_out]

                for n, l in enumerate(fc_names):
                    fc_in = ch_out if self.mul_label_ref else self.sh_fix * self.sw_fix
                    fc_layer = [sn(nn.Linear(fc_in, ch_out))]
                    for k in range(1, n_fc_layers): 
                        fc_layer += [sn(nn.Linear(ch_out, ch_out))]
                    fc_layer += [sn(nn.Linear(ch_out, fc_outs[n]))]
                    setattr(self, '%s_%d' % (l, i), nn.Sequential(*fc_layer))
                     
        ### label embedding network 
        self.label_embedding = LabelEmbedder(opt, input_nc, opt.netS, 
            params_free_layers=(self.n_adaptive_layers if self.adap_embed else 0))
            
        ### main branch layers
        for i in reversed(range(n_downsample_G + 1)):
            setattr(self, 'up_%d' % i, SPADEResnetBlock(ch[i+1], ch[i], norm=norm, hidden_nc=ch_hidden[i], 
                    conv_ks=conv_ks, spade_ks=spade_ks,
                    conv_params_free=(self.adap_conv and i < self.n_adaptive_layers),
                    norm_params_free=(self.adap_spade and i < self.n_adaptive_layers)))
                   
        self.conv_img = nn.Conv2d(nf, 3, kernel_size=3, padding=1)
        self.up = functools.partial(F.interpolate, scale_factor=2)


        ### for multiple reference images
        if opt.n_shot > 1:
            self.atn_query_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
            self.atn_key_first = SPADEConv2d(input_nc, nf, norm=norm_ref)
            for i in range(n_downsample_A):
                f_in, f_out = ch[i], ch[i+1]
                setattr(self, 'atn_key_%d' % i, SPADEConv2d(f_in, f_out, stride=2, norm=norm_ref))
                setattr(self, 'atn_query_%d' % i, SPADEConv2d(f_in, f_out, stride=2, norm=norm_ref))

        ### kld loss
        self.use_kld = opt.lambda_kld > 0
        self.z_dim = 256    
        if self.use_kld:
            f_in = min(nf_max, nf*(2**n_downsample_G)) * self.sh * self.sw
            f_out = min(nf_max, nf*(2**n_downsample_G)) * self.sh * self.sw
            self.fc_mu_ref = nn.Linear(f_in, self.z_dim)
            self.fc_var_ref = nn.Linear(f_in, self.z_dim)
            self.fc = nn.Linear(self.z_dim, f_out)    

        ### flow        
        self.warp_prev = False                            # whether to warp prev image (set when starting training multiple frames)
        self.warp_ref = opt.warp_ref and not opt.for_face # whether to warp reference image and combine with the synthesized
        if self.warp_ref:
            self.flow_network_ref = FlowGenerator(opt, 2)
            if self.spade_combine:            
                self.img_ref_embedding = LabelEmbedder(opt, opt.output_nc + 1, opt.sc_arch)

    ### when starting training multiple frames, initialize the flow network
    def set_flow_prev(self):        
        opt = self.opt
        self.warp_prev = True        
        self.sep_prev_flownet = opt.sep_flow_prev or (opt.n_frames_G != 2) or not opt.warp_ref
        self.sep_prev_embedding = self.spade_combine and (opt.sep_warp_embed or not opt.warp_ref)
        self.flow_network_temp = FlowGenerator(opt, opt.n_frames_G) if self.sep_prev_flownet else self.flow_network_ref
        if self.spade_combine:
            self.add_raw_loss = True
            self.img_prev_embedding = LabelEmbedder(opt, opt.output_nc + 1, opt.sc_arch) \
                if self.sep_prev_embedding else self.img_ref_embedding
        if self.warp_ref:
            if self.sep_prev_flownet: self.load_pretrained_net(self.flow_network_ref, self.flow_network_temp)
            if self.sep_prev_embedding: self.load_pretrained_net(self.img_ref_embedding, self.img_prev_embedding)
            self.flow_temp_is_initalized = True

    def forward(self, label, label_refs, img_refs, prev=[None, None], t=0, img_coarse=None):
        ### for face refinement
        if img_coarse is not None:
            return self.forward_face(label, label_refs, img_refs, img_coarse)        

        ### SPADE weight generation
        x, encoded_label, conv_weights, norm_weights, mu, logvar, atn, ref_idx \
            = self.weight_generation(img_refs, label_refs, label, t=t)        

        ### flow estimation
        has_prev = prev[0] is not None        
        label_ref, img_ref = self.pick_ref([label_refs, img_refs], ref_idx)
        label_prev, img_prev = prev
        flow, weight, img_warp, ds_ref = self.flow_generation(label, label_ref, img_ref, label_prev, img_prev, has_prev)

        weight_ref, weight_prev = weight
        img_ref_warp, img_prev_warp = img_warp           
        if self.add_raw_loss: encoded_label_raw = [encoded_label[i] for i in range(self.n_sc_layers)]            
        encoded_label = self.SPADE_combine(encoded_label, ds_ref)          
        
        ### main branch convolution layers
        for i in range(self.n_downsample_G, -1, -1):            
            conv_weight = conv_weights[i] if (self.adap_conv and i < self.n_adaptive_layers) else None
            norm_weight = norm_weights[i] if (self.adap_spade and i < self.n_adaptive_layers) else None                  
            if self.add_raw_loss and i < self.n_sc_layers:
                if i == self.n_sc_layers - 1: x_raw = x
                x_raw = getattr(self, 'up_'+str(i))(x_raw, encoded_label_raw[i], conv_weights=conv_weight, norm_weights=norm_weight)    
                if i != 0: x_raw = self.up(x_raw)
            x = getattr(self, 'up_'+str(i))(x, encoded_label[i], conv_weights=conv_weight, norm_weights=norm_weight)            
            if i != 0: x = self.up(x)

        ### raw synthesized image
        x = self.conv_img(actvn(x))
        img_raw = torch.tanh(x)        

        ### combine with reference / previous images
        if not self.spade_combine:
            ### combine raw result with reference image
            if self.warp_ref:
                img_final = img_raw * weight_ref + img_ref_warp * (1 - weight_ref)        
            else:
                img_final = img_raw
                if not self.warp_prev: img_raw = None

            ### combine generated frame with previous frame
            if self.warp_prev and has_prev:
                img_final = img_final * weight_prev + img_prev_warp * (1 - weight_prev)        
        else:
            img_final = img_raw
            img_raw = None if not self.add_raw_loss else torch.tanh(self.conv_img(actvn(x_raw)))
                
        return img_final, flow, weight, img_raw, img_warp, mu, logvar, atn, ref_idx

    ### forward for face refinement
    def forward_face(self, label, label_refs, img_refs, img_coarse):                
        x, encoded_label, _, norm_weights, _, _, _, _ = self.weight_generation(img_refs, label_refs, label, img_coarse=img_coarse)
        
        for i in range(self.n_downsample_G, -1, -1):            
            norm_weight = norm_weights[i] if (self.adap_spade and i < self.n_adaptive_layers) else None                  
            x = getattr(self, 'up_'+str(i))(x, encoded_label[i], norm_weights=norm_weight)            
            if i != 0: x = self.up(x)                
        
        x = self.conv_img(actvn(x))
        img_final = torch.tanh(x)        
        return img_final

    ### adaptively generate weights for SPADE in layer i of generator
    def get_SPADE_weights(self, x, i):
        if not self.mul_label_ref:
            # get fixed output size for fc layers
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)

        ch_in, ch_out = self.ch[i], self.ch[i+1]
        ch_h = self.ch_hidden[i][0]
        eks, sks = self.embed_ks, self.spade_ks

        b = x.size()[0]
        x = self.reshape_embed_input(x)
                      
        # weights for the label embedding network  
        embedding_weights = None
        if self.adap_embed:
            fc_e = getattr(self, 'fc_spade_e_'+str(i))(x).view(b, -1)
            embedding_weights = self.reshape_weight(fc_e, [ch_out, ch_in, eks, eks])

        # weights for the 3 layers in SPADE module: conv_0, conv_1, and shortcut
        fc_0 = getattr(self, 'fc_spade_0_'+str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_spade_1_'+str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_spade_s_'+str(i))(x).view(b, -1)
        weight_0 = self.reshape_weight(fc_0, [[ch_out, ch_h, sks, sks]]*2)
        weight_1 = self.reshape_weight(fc_1, [[ch_in, ch_h, sks, sks]]*2)
        weight_s = self.reshape_weight(fc_s, [[ch_out, ch_h, sks, sks]]*2)
        norm_weights = [weight_0, weight_1, weight_s]
        
        return embedding_weights, norm_weights

    ### adaptively generate weights for layer i in main branch convolutions
    def get_conv_weights(self, x, i):
        if not self.mul_label_ref:            
            x = nn.AdaptiveAvgPool2d((self.sh_fix, self.sw_fix))(x)
        ch_in, ch_out = self.ch[i], self.ch[i+1]        
        b = x.size()[0]
        x = self.reshape_embed_input(x)           
        
        fc_0 = getattr(self, 'fc_conv_0_'+str(i))(x).view(b, -1)
        fc_1 = getattr(self, 'fc_conv_1_'+str(i))(x).view(b, -1)
        fc_s = getattr(self, 'fc_conv_s_'+str(i))(x).view(b, -1)
        weight_0 = self.reshape_weight(fc_0, [ch_in, ch_out, 3, 3])
        weight_1 = self.reshape_weight(fc_1, [ch_in, ch_in, 3, 3])
        weight_s = self.reshape_weight(fc_s, [ch_in, ch_out, 1, 1])
        return [weight_0, weight_1, weight_s]

    ### attention to combine in the case of multiple reference images
    def attention_module(self, x, label, label_ref, attention=None):
        b, c, h, w = x.size()
        n = self.opt.n_shot
        b = b//n

        if attention is None:
            atn_key = self.atn_key_first(label_ref)
            atn_query = self.atn_query_first(label)

            for i in range(self.n_downsample_A):
                atn_key = getattr(self, 'atn_key_' + str(i))(atn_key)
                atn_query = getattr(self, 'atn_query_' + str(i))(atn_query)

            atn_key = atn_key.view(b, n, c, -1).permute(0, 1, 3, 2).contiguous().view(b, -1, c)  # B X NHW X C
            atn_query = atn_query.view(b, c, -1)  # B X C X HW
            energy = torch.bmm(atn_key, atn_query)  # B X NHW X HW
            attention = nn.Softmax(dim=1)(energy)

        x = x.view(b, n, c, h * w).permute(0, 2, 1, 3).contiguous().view(b, c, -1)  # B X C X NHW
        out = torch.bmm(x, attention).view(b, c, h, w)

        atn_vis = attention.view(b, n, h * w, h * w).sum(2).view(b, n, h, w)  # B X N X HW
        return out, attention, atn_vis[-1:, 0:1]      

    ### pick the reference image that is most similar to current frame
    def pick_ref(self, refs, ref_idx):
        if type(refs) == list:
            return [self.pick_ref(r, ref_idx) for r in refs]
        if ref_idx is None:
            return refs[:,0]
        ref_idx = ref_idx.long().view(-1, 1, 1, 1, 1)
        ref = refs.gather(1, ref_idx.expand_as(refs)[:,0:1])[:,0]        
        return ref

    ### compute kld loss at the bottleneck of generator
    def compute_kld(self, x, label, img_coarse):
        mu = logvar = None
        if img_coarse is not None:
            if self.concat_label_ref: img_coarse = torch.cat([img_coarse, label], dim=1)
            x_kld = self.ref_img_first(img_coarse)
            for i in range(self.n_downsample_G):
                x_kld = getattr(self, 'ref_img_down_'+str(i))(x_kld)

        elif self.use_kld:
            b, c, h, w = x.size()            
            mu = self.fc_mu_ref(x.view(b, -1))
            if self.opt.isTrain:
                logvar = self.fc_var_ref(x.view(b, -1))
                z = self.reparameterize(mu, logvar)                
            else:
                z = mu
            x_kld = self.fc(z).view(b, -1, h, w)            
        else:
            x_kld = x
        return x_kld, mu, logvar

    ### encode the reference image to get features for weight generation
    def reference_encoding(self, img_ref, label_ref, label, n, t=0):
        if self.concat_label_ref: 
            # concat reference label map and image together for encoding
            concat_ref = torch.cat([img_ref, label_ref], dim=1)
            x = self.ref_img_first(concat_ref)
        elif self.mul_label_ref:
            # apply conv to both reference label and image, then multiply them together for encoding
            x = self.ref_img_first(img_ref)
            x_label = self.ref_label_first(label_ref)
        else:
            assert False

        atn_vis = ref_idx = None # attention map and the index of the most similar reference image
        for i in range(self.n_downsample_G):            
            x = getattr(self, 'ref_img_down_'+str(i))(x)
            if self.mul_label_ref: 
                x_label = getattr(self, 'ref_label_down_'+str(i))(x_label)

            ### combine different reference images at a particular layer if n_shot > 1
            if n > 1 and i == self.n_downsample_A - 1:
                x, atn, atn_vis = self.attention_module(x, label, label_ref)
                if self.mul_label_ref:
                    x_label, _, _ = self.attention_module(x_label, None, None, atn)

                atn_sum = atn.view(label.shape[0], n, -1).sum(2)
                ref_idx = torch.argmax(atn_sum, dim=1)                
        
        # get all corresponding layers in the encoder output for generating weights in corresponding layers
        encoded_ref = None
        if self.opt.isTrain or n > 1 or t == 0:
            encoded_image_ref = [x]   
            if self.mul_label_ref: encoded_label_ref = [x_label]       
            
            for i in reversed(range(self.n_downsample_G)):
                conv = getattr(self, 'ref_img_up_'+str(i))(encoded_image_ref[-1])
                encoded_image_ref.append(conv)
                if self.mul_label_ref:
                    conv_label = getattr(self, 'ref_label_up_'+str(i))(encoded_label_ref[-1])            
                    encoded_label_ref.append(conv_label)
            
            if self.mul_label_ref:
                encoded_ref = []
                for i in range(len(encoded_image_ref)):  
                    conv, conv_label = encoded_image_ref[i], encoded_label_ref[i]
                    b, c, h, w = conv.size()
                    conv_label = nn.Softmax(dim=1)(conv_label)        
                    conv_prod = (conv.view(b, c, 1, h*w) * conv_label.view(b, 1, c, h*w)).sum(3, keepdim=True)                    
                    encoded_ref.append(conv_prod)
            else:
                encoded_ref = encoded_image_ref
            encoded_ref = encoded_ref[::-1]

        return x, encoded_ref, atn_vis, ref_idx

    ### generate weights based on the encoded features
    def weight_generation(self, img_ref, label_ref, label, t=0, img_coarse=None):
        b, n, c, h, w = img_ref.size()
        img_ref, label_ref = img_ref.view(b*n, -1, h, w), label_ref.view(b*n, -1, h, w)                               

        x, encoded_ref, atn, ref_idx = self.reference_encoding(img_ref, label_ref, label, n, t)
        x_kld, mu, logvar = self.compute_kld(x, label, img_coarse)
        
        if self.opt.isTrain or n > 1 or t == 0:
            embedding_weights, norm_weights, conv_weights = [], [], []            
            for i in range(self.n_adaptive_layers):                                
                if self.adap_spade:       
                    feat = encoded_ref[min(len(encoded_ref)-1, i+1)]                         
                    embedding_weight, norm_weight = self.get_SPADE_weights(feat, i)
                    embedding_weights.append(embedding_weight)
                    norm_weights.append(norm_weight)  
                if self.adap_conv:                
                    feat = encoded_ref[min(len(encoded_ref)-1, i)]
                    conv_weights.append(self.get_conv_weights(feat, i))

            if not self.opt.isTrain:
                self.embedding_weights, self.conv_weights, self.norm_weights = embedding_weights, conv_weights, norm_weights
        else:
            embedding_weights, conv_weights, norm_weights = self.embedding_weights, self.conv_weights, self.norm_weights
        
        encoded_label = self.label_embedding(label, weights=(embedding_weights if self.adap_embed else None))        

        return x_kld, encoded_label, conv_weights, norm_weights, mu, logvar, atn, ref_idx

    def flow_generation(self, label, label_ref, img_ref, label_prev, img_prev, has_prev):
        flow, weight, img_warp, ds_ref = [None] * 2, [None] * 2, [None] * 2, [None] * 2
        if self.warp_ref:
            flow_ref, weight_ref = self.flow_network_ref(label, label_ref, img_ref, for_ref=True)
            img_ref_warp = self.resample(img_ref, flow_ref)
            flow[0], weight[0], img_warp[0] = flow_ref, weight_ref, img_ref_warp[:,:3]

        if self.warp_prev and has_prev:
            flow_prev, weight_prev = self.flow_network_temp(label, label_prev, img_prev)
            img_prev_warp = self.resample(img_prev[:,-3:], flow_prev)            
            flow[1], weight[1], img_warp[1] = flow_prev, weight_prev, img_prev_warp        

        if self.spade_combine:
            if self.warp_ref:
                ds_ref[0] = torch.cat([img_ref_warp, weight_ref], dim=1)
            if self.warp_prev and has_prev: 
                ds_ref[1] = torch.cat([img_prev_warp, weight_prev], dim=1)

        return flow, weight, img_warp, ds_ref    

    ### if using SPADE for combination
    def SPADE_combine(self, encoded_label, ds_ref):        
        if self.spade_combine:            
            encoded_image_warp = [self.img_ref_embedding(ds_ref[0]),
                                  self.img_prev_embedding(ds_ref[1]) if ds_ref[1] is not None else None]
            for i in range(self.n_sc_layers):
                encoded_label[i] = [encoded_label[i]] + [w[i] if w is not None else None for w in encoded_image_warp]
        return encoded_label

class FlowGenerator(BaseNetwork):
    def __init__(self, opt, n_frames_G):
        super().__init__()
        self.opt = opt                
        input_nc = (opt.label_nc if opt.label_nc != 0 else opt.input_nc) * n_frames_G
        input_nc += opt.output_nc * (n_frames_G - 1)        
        nf = opt.nff
        n_blocks = opt.n_blocks_F
        n_downsample_F = opt.n_downsample_F
        self.flow_multiplier = opt.flow_multiplier        
        nf_max = 1024
        ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_F + 1)]
                
        norm = opt.norm_F
        norm_layer = get_nonspade_norm_layer(opt, norm)
        activation = nn.LeakyReLU(0.2, True)
        
        down_flow = [norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1)), activation]        
        for i in range(n_downsample_F):            
            down_flow += [norm_layer(nn.Conv2d(ch[i], ch[i+1], kernel_size=3, padding=1, stride=2)), activation]            
                   
        ### resnet blocks
        res_flow = []
        ch_r = min(nf_max, nf * (2**n_downsample_F))        
        for i in range(n_blocks):
            res_flow += [SPADEResnetBlock(ch_r, ch_r, norm=norm)]
    
        ### upsample
        up_flow = []                         
        for i in reversed(range(n_downsample_F)):
            if opt.flow_deconv:
                up_flow += [norm_layer(nn.ConvTranspose2d(ch[i+1], ch[i], kernel_size=3, stride=2, padding=1, output_padding=1)), activation]
            else:
                up_flow += [nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(ch[i+1], ch[i], kernel_size=3, padding=1)), activation]
                              
        conv_flow = [nn.Conv2d(nf, 2, kernel_size=3, padding=1)]
        conv_w = [nn.Conv2d(nf, 1, kernel_size=3, padding=1), nn.Sigmoid()] 
      
        self.down_flow = nn.Sequential(*down_flow)        
        self.res_flow = nn.Sequential(*res_flow)                                            
        self.up_flow = nn.Sequential(*up_flow)
        self.conv_flow = nn.Sequential(*conv_flow)        
        self.conv_w = nn.Sequential(*conv_w)

    def forward(self, label, label_prev, img_prev, for_ref=False):
        label = torch.cat([label, label_prev, img_prev], dim=1)        
        downsample = self.down_flow(label)
        res = self.res_flow(downsample)
        flow_feat = self.up_flow(res)        
        flow = self.conv_flow(flow_feat) * self.flow_multiplier
        weight = self.conv_w(flow_feat)
        return flow, weight

class LabelEmbedder(BaseNetwork):
    def __init__(self, opt, input_nc, netS=None, params_free_layers=0, first_layer_free=False):
        super().__init__()        
        self.opt = opt        
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_F)
        activation = nn.LeakyReLU(0.2, True)        
        nf = opt.ngf
        nf_max = 1024
        self.netS = netS if netS is not None else opt.netS
        self.unet = 'unet' in self.netS
        self.decode = 'decoder' in self.netS or self.unet
        self.n_downsample_S = n_downsample_S = opt.n_downsample_G        
        self.params_free_layers = params_free_layers if params_free_layers != -1 else n_downsample_S
        self.first_layer_free = first_layer_free    
        ch = [min(nf_max, nf * (2 ** i)) for i in range(n_downsample_S + 1)]
       
        if not first_layer_free:
            layer = [nn.Conv2d(input_nc, nf, kernel_size=3, padding=1), activation]
            self.conv_first = nn.Sequential(*layer)
        
        # downsample
        for i in range(n_downsample_S):            
            layer = [nn.Conv2d(ch[i], ch[i+1], kernel_size=3, stride=2, padding=1), activation]
            if i >= params_free_layers or 'decoder' in netS:
                setattr(self, 'down_%d' % i, nn.Sequential(*layer))

        # upsample
        if self.decode:
            for i in reversed(range(n_downsample_S)):
                ch_i = ch[i+1] * (2 if self.unet and i != n_downsample_S -1 else 1)                
                layer = [nn.ConvTranspose2d(ch_i, ch[i], kernel_size=3, stride=2, padding=1, output_padding=1), activation]                
                if i >= params_free_layers:
                    setattr(self, 'up_%d' % i, nn.Sequential(*layer))                

    def forward(self, input, weights=None):
        if input is None: return None
        if self.first_layer_free:
            output = [batch_conv(input, weights[0])]
            weights = weights[1:]
        else:
            output = [self.conv_first(input)]
        for i in range(self.n_downsample_S):
            if i >= self.params_free_layers or self.decode:                
                conv = getattr(self, 'down_%d' % i)(output[-1])
            else:                
                conv = batch_conv(output[-1], weights[i], stride=2)
            output.append(conv)

        if not self.decode:
            return output

        if not self.unet:
            output = [output[-1]]
        for i in reversed(range(self.n_downsample_S)):
            input_i = output[-1]
            if self.unet and i != self.n_downsample_S-1:
                input_i = torch.cat([input_i, output[i+1]], dim=1)
            if i >= self.params_free_layers:                
                conv = getattr(self, 'up_%d' % i)(input_i)
            else:                
                conv = batch_conv(input_i, weights[i], stride=0.5)
            output.append(conv)
        if self.unet:
            output = output[self.n_downsample_S:]   
        return output[::-1]