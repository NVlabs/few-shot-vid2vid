# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os.path as path
import glob
import torch
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_img_params, get_video_params, get_transform
from data.image_folder import make_dataset, make_grouped_dataset, check_path_valid

class FewshotStreetDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataroot='datasets/street/')
        parser.add_argument('--label_nc', type=int, default=20, help='# of input label channels')      
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')        
        parser.add_argument('--aspect_ratio', type=float, default=2)         
        parser.set_defaults(resize_or_crop='random_scale_and_crop')
        parser.set_defaults(niter=20)
        parser.set_defaults(niter_single=10)
        parser.set_defaults(niter_step=2)
        parser.set_defaults(save_epoch_freq=1)        

        ### for inference        
        parser.add_argument('--seq_path', type=str, default='datasets/street/test_images/01/', help='path to the driving sequence')        
        parser.add_argument('--ref_img_path', type=str, default='datasets/street/test_images/02/', help='path to the reference image')
        parser.add_argument('--ref_img_id', type=str, default='0', help='indices of reference frames')
        return parser

    def initialize(self, opt):
        self.opt = opt
        root = opt.dataroot       
        self.L_is_label = self.opt.label_nc != 0 

        if opt.isTrain:
            self.L_paths = sorted(make_grouped_dataset(path.join(root, 'train_labels'))) 
            self.I_paths = sorted(make_grouped_dataset(path.join(root, 'train_images')))
            check_path_valid(self.L_paths, self.I_paths)

            self.n_of_seqs = len(self.L_paths)       
            print('%d sequences' % self.n_of_seqs)          
        else:
            self.I_paths = sorted(make_dataset(opt.seq_path))            
            self.L_paths = sorted(make_dataset(opt.seq_path.replace('images', 'labels')))
            self.ref_I_paths = sorted(make_dataset(opt.ref_img_path))
            self.ref_L_paths = sorted(make_dataset(opt.ref_img_path.replace('images', 'labels')))

    def __getitem__(self, index):    
        opt = self.opt        
        if opt.isTrain:
            L_paths = self.L_paths[index % self.n_of_seqs]
            I_paths = self.I_paths[index % self.n_of_seqs]
            ref_L_paths, ref_I_paths = L_paths, I_paths            
        else:
            L_paths, I_paths = self.L_paths, self.I_paths
            ref_L_paths, ref_I_paths = self.ref_L_paths, self.ref_I_paths
        
        
        ### setting parameters                
        n_frames_total, start_idx, t_step, ref_indices = get_video_params(opt, self.n_frames_total, len(I_paths), index)        
        w, h = opt.fineSize, int(opt.fineSize / opt.aspect_ratio)
        img_params = get_img_params(opt, (w, h))
        is_first_frame = opt.isTrain or index == 0

        transform_I = get_transform(opt, img_params, color_aug=opt.isTrain)
        transform_L = get_transform(opt, img_params, method=Image.NEAREST, normalize=False) if self.L_is_label else transform_I


        ### read in reference image
        Lr, Ir = self.Lr, self.Ir
        if is_first_frame:            
            for idx in ref_indices:                
                Li = self.get_image(ref_L_paths[idx], transform_L, is_label=self.L_is_label)            
                Ii = self.get_image(ref_I_paths[idx], transform_I)
                Lr = self.concat_frame(Lr, Li.unsqueeze(0))
                Ir = self.concat_frame(Ir, Ii.unsqueeze(0))

            if not opt.isTrain: # keep track of non-changing variables during inference                
                self.Lr, self.Ir = Lr, Ir


        ### read in target images
        L, I = self.L, self.I
        for t in range(n_frames_total):
            idx = start_idx + t * t_step            
            Lt = self.get_image(L_paths[idx], transform_L, is_label=self.L_is_label)
            It = self.get_image(I_paths[idx], transform_I)
            L = self.concat_frame(L, Lt.unsqueeze(0))
            I = self.concat_frame(I, It.unsqueeze(0))
            
        if not opt.isTrain:
            self.L, self.I = L, I
        
        seq = path.basename(path.dirname(opt.ref_img_path)) + '-' + opt.ref_img_id + '_' + path.basename(path.dirname(opt.seq_path))
        
        return_list = {'tgt_label': L, 'tgt_image': I, 'ref_label': Lr, 'ref_image': Ir,
                       'path': I_paths[idx], 'seq': seq}
        return return_list

    def get_image(self, A_path, transform_scaleA, is_label=False):
        if is_label: return self.get_label_tensor(A_path, transform_scaleA)        
        A_img = self.read_data(A_path)
        A_scaled = transform_scaleA(A_img)        
        return A_scaled

    def get_label_tensor(self, label_path, transform_label):
        label = self.read_data(label_path).convert('L')
        
        train2eval = self.opt.label_nc == 20
        if train2eval:            
            ### 35 to 20            
            A_label_np = np.array(label)
            label_mapping = np.array([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3, 4, 19, 19, 19, 5, 19, 
                                      6, 7, 8, 9, 18, 10, 11, 12, 13, 14, 19, 19, 15, 16, 17, 19], dtype=np.uint8)            
            A_label_np = label_mapping[A_label_np]
            label = Image.fromarray(A_label_np)

        label_tensor = transform_label(label) * 255.0 
        return label_tensor

    def __len__(self):
        if not self.opt.isTrain: return len(self.L_paths)
        return max(10000, sum([len(L) for L in self.L_paths]))

    def name(self):
        return 'StreetDataset'