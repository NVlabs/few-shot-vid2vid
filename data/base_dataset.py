### Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
### Licensed under the Nvidia Source Code License.

from PIL import Image
import numpy as np
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from util.distributed import master_only_print as print

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.L = self.I = self.Lr = self.Ir = None
        self.n_frames_total = 1  # current number of frames to train in a single iteration

    def name(self):
        return 'BaseDataset'

    def update_training_batch(self, ratio):
        # update the training sequence length to be longer        
        seq_len_max = 30
        if self.n_frames_total < seq_len_max:
            self.n_frames_total = min(seq_len_max, self.opt.n_frames_total * (2**ratio))
            print('--- Updating training sequence length to %d ---' % self.n_frames_total)

    def read_data(self, path, data_type='img'):
        is_img = data_type == 'img'
        if is_img:
            img = Image.open(path)
        elif data_type == 'np':
            img = np.loadtxt(path, delimiter=',')
        else:
            img = path
        return img

    def crop(self, img, coords):
        min_y, max_y, min_x, max_x = coords
        if isinstance(img, np.ndarray):
            return img[min_y:max_y, min_x:max_x]
        else:
            return img.crop((min_x, min_y, max_x, max_y))

    def concat_frame(self, A, Ai):
        if not self.opt.isTrain:
            return Ai

        if A is None:
            A = Ai
        else:
            A = torch.cat([A, Ai])
        return A

def get_img_params(opt, size):
    w, h = size
    new_w, new_h = w, h
    
    # resize input image
    if 'resize' in opt.resize_or_crop:
        new_h = new_w = opt.loadSize
    else:
        if 'scale_width' in opt.resize_or_crop:
            new_w = opt.loadSize             
        elif 'random_scale' in opt.resize_or_crop:
            new_w = random.randint(int(opt.fineSize), int(1.2*opt.fineSize))
        new_h = int(new_w * h) // w      
    if 'crop' not in opt.resize_or_crop:
        new_h = int(new_w // opt.aspect_ratio)
    new_w = new_w // 4 * 4
    new_h = new_h // 4 * 4  
         
    # crop resized image
    size_x = min(opt.loadSize, opt.fineSize)
    size_y = size_x // opt.aspect_ratio
    if not opt.isTrain: # crop central region
        pos_x = (new_w - size_x) // 2
        pos_y = (new_h - size_y) // 2    
    else:               # crop random region
        pos_x = random.randint(0, np.maximum(0, new_w - size_x))
        pos_y = random.randint(0, np.maximum(0, new_h - size_y))        

    # for color augmentation
    h_b = random.uniform(-30, 30)
    s_a = random.uniform(0.8, 1.2)
    s_b = random.uniform(-10, 10)
    v_a = random.uniform(0.8, 1.2)
    v_b = random.uniform(-10, 10)    
    
    flip = random.random() > 0.5
    return {'new_size': (new_w, new_h), 'crop_pos': (pos_x, pos_y), 'crop_size': (size_x, size_y), 'flip': flip, 
            'color_aug': (h_b, s_a, s_b, v_a, v_b)}

def get_video_params(opt, n_frames_total, cur_seq_len, index):    
    if opt.isTrain:                
        n_frames_total = min(cur_seq_len, n_frames_total)             # total number of frames to load
        max_t_step = min(opt.max_t_step, (cur_seq_len-1) // max(1, (n_frames_total-1)))        
        t_step = np.random.randint(max_t_step) + 1                    # spacing between neighboring sampled frames                
        
        offset_max = max(1, cur_seq_len - (n_frames_total-1)*t_step)  # maximum possible frame index for the first frame
        if 'pose' in opt.dataset_mode:
            start_idx = index % offset_max                            # offset for the first frame to load
            max_range, min_range = 60, 14                             # range for possible reference frames
        else:
            start_idx = np.random.randint(offset_max)                 # offset for the first frame to load        
            max_range, min_range = 300, 14                            # range for possible reference frames
        
        ref_range = list(range(max(0, start_idx - max_range), max(1, start_idx - min_range))) \
                  + list(range(min(start_idx + min_range, cur_seq_len - 1), min(start_idx + max_range, cur_seq_len)))
        ref_indices = np.random.choice(ref_range, size=opt.n_shot)    # indices for reference frames

    else:
        n_frames_total = 1
        start_idx = index
        t_step = 1        
        ref_indices = opt.ref_img_id.split(',')
        ref_indices = [int(i) for i in ref_indices]        

    return n_frames_total, start_idx, t_step, ref_indices

def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True, color_aug=False):
    transform_list = []    
    transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], params['crop_size'])))

    if opt.isTrain and color_aug:
        transform_list.append(transforms.Lambda(lambda img: __color_aug(img, params['color_aug'])))    

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size    
    return img.crop((x1, y1, x1 + tw, y1 + th))    

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __color_aug(img, params):
    h, s, v = img.convert('HSV').split()    
    h = h.point(lambda i: (i + params[0]) % 256)
    s = s.point(lambda i: min(255, max(0, i * params[1] + params[2])))
    v = v.point(lambda i: min(255, max(0, i * params[3] + params[4])))
    img = Image.merge('HSV', (h, s, v)).convert('RGB')
    return img