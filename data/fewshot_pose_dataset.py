# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os.path as path
import torch
from PIL import Image
import numpy as np
import random
import json

from data.base_dataset import BaseDataset, get_img_params, get_video_params, get_transform
from data.image_folder import make_dataset, make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints
from util.distributed import master_only_print as print

class FewshotPoseDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(dataroot='datasets/pose/')
        parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        parser.add_argument('--input_nc', type=int, default=6, help='# of input image channels')        
        parser.add_argument('--aspect_ratio', type=float, default=0.5)
        parser.add_argument('--pose_type', type=str, default='both', help='only both is supported now')
        parser.add_argument('--remove_face_labels', action='store_true', help='remove face part of labels to prevent overfitting')         
        parser.add_argument('--refine_face', action='store_true', help='if specified, refine face region with an additional generator')  
        parser.add_argument('--basic_point_only', action='store_true', help='only use basic joints without face and hand')            

        ### for inference        
        parser.add_argument('--seq_path', type=str, default='datasets/pose/test_images/01/', help='path to the driving sequence')        
        parser.add_argument('--ref_img_path', type=str, default='datasets/pose/test_images/02/', help='path to the reference image')
        parser.add_argument('--ref_img_id', type=str, default='0', help='indices of reference frames')   
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.pose_type = opt.pose_type
        
        root = opt.dataroot 
        if opt.isTrain:
            self.img_paths = sorted(make_grouped_dataset(path.join(root, 'train_images')))
            self.op_paths = sorted(make_grouped_dataset(path.join(root, 'train_openpose')))
            self.dp_paths = sorted(make_grouped_dataset(path.join(root, 'train_densepose')))
            self.ppl_indices = None
            if path.exists(path.join(root, 'all_subsequences.json')):
                with open(path.join(root, 'all_subsequences.json')) as f:
                    all_subsequences = json.loads(f.read())
                seq_indices = all_subsequences['seq_indices']
                start_frame_indices = all_subsequences['start_frame_indices']
                end_frame_indices = all_subsequences['end_frame_indices']
                img_paths, op_paths, dp_paths = [], [], []
                for i in range(len(seq_indices)):
                    seq_idx = seq_indices[i]
                    start_frame_idx, end_frame_idx = start_frame_indices[i], end_frame_indices[i]
                    img_paths += [self.img_paths[seq_idx][start_frame_idx : end_frame_idx]]
                    op_paths += [self.op_paths[seq_idx][start_frame_idx: end_frame_idx]]
                    dp_paths += [self.dp_paths[seq_idx][start_frame_idx: end_frame_idx]]
                self.img_paths = img_paths
                self.op_paths = op_paths
                self.dp_paths = dp_paths
                self.ppl_indices = all_subsequences['ppl_indices']
        else:    
            self.img_paths = sorted(make_dataset(opt.seq_path))
            self.op_paths = sorted(make_dataset(opt.seq_path.replace('images', 'openpose')))
            self.dp_paths = sorted(make_dataset(opt.seq_path.replace('images', 'densepose')))

            self.ref_img_paths = sorted(make_dataset(opt.ref_img_path))
            self.ref_op_paths = sorted(make_dataset(opt.ref_img_path.replace('images', 'openpose')))
            self.ref_dp_paths = sorted(make_dataset(opt.ref_img_path.replace('images', 'densepose')))

        self.n_of_seqs = len(self.img_paths)        # number of sequences to train 
        if opt.isTrain: print('%d sequences' % self.n_of_seqs)        
        self.crop_coords = self.ref_face_pts = None
        self.ref_crop_coords = [None] * opt.n_shot

    def __getitem__(self, index):
        opt = self.opt
        if opt.isTrain:
            # np.random.seed(index)
            seq_idx = random.randrange(self.n_of_seqs) # which sequence to load

            img_paths = self.img_paths[seq_idx]
            op_paths = self.op_paths[seq_idx]
            dp_paths = self.dp_paths[seq_idx]
            ppl_indices = self.ppl_indices[seq_idx] if self.ppl_indices is not None else None
            ref_img_paths, ref_op_paths, ref_dp_paths, ref_ppl_indices = img_paths, op_paths, dp_paths, ppl_indices
        else:            
            img_paths, op_paths, dp_paths = self.img_paths, self.op_paths, self.dp_paths
            ref_img_paths, ref_op_paths, ref_dp_paths = self.ref_img_paths, self.ref_op_paths, self.ref_dp_paths
            ppl_indices = ref_ppl_indices = None
        
        ### setting parameters
        # n_frames_total: # of frames to train
        # start_idx: which frame index to start with
        # t_step: # of frames between neighboring frames
        # ref_indices: frame indices for the reference images
        n_frames_total, start_idx, t_step, ref_indices = get_video_params(self.opt, self.n_frames_total, len(img_paths), index)
        w, h = opt.fineSize, int(opt.fineSize / opt.aspect_ratio)
        img_params = get_img_params(opt, (w, h))
        is_first_frame = opt.isTrain or index == 0        

        ### reference image
        Lr, Ir = self.Lr, self.Ir         
        if is_first_frame: # need to read reference images for every training iter or at beginning of inference            
            ref_crop_coords = [None] * opt.n_shot
            for i, idx in enumerate(ref_indices):                
                ref_size = self.read_data(ref_img_paths[idx]).size
                Li, Ii, ref_crop_coords[i], ref_face_pts = self.get_images(ref_img_paths, ref_op_paths, ref_dp_paths,
                    ref_ppl_indices, idx, ref_size, img_params, self.ref_crop_coords[i])
                Lr = self.concat_frame(Lr, Li.unsqueeze(0))            
                Ir = self.concat_frame(Ir, Ii.unsqueeze(0))                            
        
            if not opt.isTrain: # keep track of non-changing variables during inference
                read_keypoints.face_ratio = None 
                self.Lr, self.Ir = Lr, Ir
                self.ref_face_pts = None
                self.ref_crop_coords = ref_crop_coords                

        ### target image        
        size = self.read_data(img_paths[0]).size
        crop_coords = self.crop_coords if not opt.isTrain else ref_crop_coords[0]        

        L, I = self.L, self.I        
        for t in range(n_frames_total):
            idx = start_idx + t * t_step
            Lt, It, crop_coords, _ = self.get_images(img_paths, op_paths, dp_paths, ppl_indices, idx, size,
                img_params, crop_coords, self.ref_face_pts)
            L = self.concat_frame(L, Lt.unsqueeze(0))
            I = self.concat_frame(I, It.unsqueeze(0))                    

        if not opt.isTrain:
            self.L, self.I = L, I                
            if index == 0: self.crop_coords = crop_coords            
        seq = path.basename(path.dirname(opt.ref_img_path)) + '-' + opt.ref_img_id + '_' + path.basename(path.dirname(opt.seq_path))

        return_list = {'tgt_label': L, 'tgt_image': I, 'ref_label': Lr, 'ref_image': Ir,
                       'path': img_paths[idx], 'seq': seq}

        return return_list

    def get_images(self, img_paths, op_paths, dp_paths, ppl_indices, i, size, params, crop_coords, ref_face_pts=None):
        img_path = img_paths[i]
        op_path = op_paths[i]
        dp_path = dp_paths[i]
        ppl_idx = ppl_indices[i] if ppl_indices is not None else None
                 
        # openpose
        O, op, crop_coords, face_pts = self.get_image(op_path, size, params, crop_coords, input_type='openpose',
                                                      ppl_idx=ppl_idx, ref_face_pts=ref_face_pts)
        # densepose
        D = self.get_image(dp_path, size, params, crop_coords, input_type='densepose', op=op)
        # concatenate both pose maps
        Li = torch.cat([D, O])

        # RGB image
        Ii = self.get_image(img_path, size, params, crop_coords, input_type='img')

        return Li, Ii, crop_coords, face_pts
    
    def get_image(self, A_path, size, params, crop_coords, input_type, ppl_idx=None, op=None, ref_face_pts=None):
        if A_path is None: return None, None
        opt = self.opt        
        is_img = input_type == 'img'
        method = Image.BICUBIC if is_img else Image.NEAREST       

        if input_type == 'openpose':
            # get image from openpose keypoints
            A_img, pose_pts, face_pts = read_keypoints(opt, A_path, size,
                opt.basic_point_only, opt.remove_face_labels, ppl_idx, ref_face_pts)

            # randomly crop the image
            A_img, crop_coords = self.crop_person_region(A_img, crop_coords, pose_pts, size)

        else:
            A_img = self.read_data(A_path)            
            A_img, _ = self.crop_person_region(A_img, crop_coords)            
            if input_type == 'densepose':  # remove other ppl in the densepose map
                A_img = self.remove_other_ppl(A_img, A_path, crop_coords, op)
        
        transform_scaleA = get_transform(self.opt, params, method=method, color_aug=is_img and opt.isTrain)
        A_scaled = transform_scaleA(A_img).float()

        if input_type == 'densepose': # renormalize the part labels
            A_scaled[2,:,:] = ((A_scaled[2,:,:] * 0.5 + 0.5) * 255 / 24 - 0.5) / 0.5

        if input_type == 'openpose':
            return A_scaled, A_img, crop_coords, face_pts
        return A_scaled      

    # only crop the person region in the image
    def crop_person_region(self, A_img, crop_coords, pose_pts=None, size=None):
        # get the crop coordinates
        if crop_coords is None: 
            offset_max = 0.05
            random_offset = [random.uniform(-offset_max, offset_max), 
                             random.uniform(-offset_max, offset_max)] if self.opt.isTrain else [0,0]
            crop_coords = self.get_crop_coords(pose_pts, size, random_offset)

        # only crop the person region
        if type(A_img) == np.ndarray:
            xs, ys, xe, ye = crop_coords
            A_img = Image.fromarray(A_img[ys:ye, xs:xe, :])
        else:
            A_img = A_img.crop(crop_coords)
        return A_img, crop_coords

    ### get the pixel coordinates to crop
    def get_crop_coords(self, pose_pts, size, offset=None):                
        w, h = size        
        valid = pose_pts[:, 0] != 0
        x, y = pose_pts[valid, 0], pose_pts[valid, 1]
        
        # get the center position and length of the person to crop
        # ylen, xlen: height and width of the person
        # y_cen, x_cen: center of the person
        x_cen = int(x.min() + x.max()) // 2 if x.shape[0] else w // 2
        if y.shape[0]:
            y_min = max(y.min(), min(pose_pts[15, 1], pose_pts[16, 1]))            
            y_max = max(pose_pts[11, 1], pose_pts[14, 1])
            if y_max == 0: y_max = y.max()
            y_cen = int(y_min + y_max) // 2 
            y_len = y_max - y_min
        else:
            y_cen = y_len = h // 2        
        
        # randomly scale the crop size for augmentation
        # final cropped size = person height * scale
        scale = random.uniform(1.4, 1.6) if self.opt.isTrain else 1.5

        # bh, bw: half of height / width of final cropped size        
        bh = int(min(h, max(h//4, y_len * scale))) // 2
        bw = int(bh * self.opt.aspect_ratio)

        # randomly offset the cropped position for augmentation
        if offset is not None:
            x_cen += int(offset[0]*bw)
            y_cen += int(offset[1]*bh)
        x_cen = max(bw, min(w-bw, x_cen))
        y_cen = max(bh, min(h-bh, y_cen))        

        return [(x_cen-bw), (y_cen-bh), (x_cen+bw), (y_cen+bh)]

    # remove other people in the densepose map by looking at the id in the densemask map
    def remove_other_ppl(self, A_img, A_path, crop_coords, op):
        B_path = A_path.replace('densepose', 'densemask').replace('IUV', 'INDS')
        if path.exists(B_path):
            B_img = self.read_data(B_path)
            B_img = np.array(B_img.crop(crop_coords))
            op = np.array(op)
            valid = ((op[:,:,0] > 0) | (op[:,:,1] > 0) | (op[:,:,2] > 0))
            dp_valid = B_img[valid]
            dp_valid = dp_valid[dp_valid != 0]
            if dp_valid.size != 0:
                inds = np.bincount(dp_valid).argmax()
                A_np = np.array(A_img)
                mask = (B_img == inds)
                if mask.ndim == 2:
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                A_np = A_np * mask
                A_img = Image.fromarray(A_np)
        return A_img

    def __len__(self):        
        if not self.opt.isTrain: return len(self.img_paths)
        return max(10000, max([len(A) for A in self.img_paths]))  # max number of frames in the training sequences

    def name(self):
        return 'PoseDataset'