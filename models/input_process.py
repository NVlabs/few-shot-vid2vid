# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch

############################# input processing ###################################
def encode_input(opt, data_list, dummy_bs):
    if opt.isTrain and data_list[0].get_device() == 0:
        data_list = remove_dummy_from_tensor(opt, data_list, dummy_bs)
    tgt_label, tgt_image, flow_gt, conf_gt, ref_label, ref_image, prev_label, prev_image = data_list

    # target label and image
    tgt_label = encode_label(opt, tgt_label)
    tgt_image = tgt_image.cuda()            
             
    # reference label and image
    ref_label = encode_label(opt, ref_label)        
    ref_image = ref_image.cuda()
        
    return tgt_label, tgt_image, flow_gt, conf_gt, ref_label, ref_image, prev_label, prev_image

def encode_label(opt, label_map):
    size = label_map.size()
    if len(size) == 5:
        bs, t, c, h, w = size
        label_map = label_map.view(-1, c, h, w)
    else:
        bs, c, h, w = size        

    label_nc = opt.label_nc
    if label_nc == 0:
        input_label = label_map.cuda()
    else:
        # create one-hot vector for label map                         
        label_map = label_map.cuda()
        oneHot_size = (label_map.shape[0], label_nc, h, w)
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.long().cuda(), 1.0)
    
    if len(size) == 5:
        return input_label.view(bs, t, -1, h, w)
    return input_label

### get the union of target and reference foreground masks
def combine_fg_mask(fg_mask, ref_fg_mask, has_fg):
    return ((fg_mask > 0) | (ref_fg_mask > 0)).float() if has_fg else 1

### obtain the foreground mask for pose sequences, which only includes the human
def get_fg_mask(opt, input_label, has_fg):
    if type(input_label) == list:
        return [get_fg_mask(opt, l, has_fg) for l in input_label]
    if not has_fg: return None
    if len(input_label.size()) == 5: input_label = input_label[:,0]            
    mask = input_label[:,2:3] if opt.label_nc == 0 else -input_label[:,0:1]
    
    mask = torch.nn.MaxPool2d(15, padding=7, stride=1)(mask) # make the mask slightly larger
    mask = (mask > -1).float()    
    return mask

### obtain mask of different body parts
def get_part_mask(pose):
    part_groups = [[0], [1,2], [3,4], [5,6], [7,9,8,10], [11,13,12,14], [15,17,16,18], [19,21,20,22], [23,24]]
    n_parts = len(part_groups)

    need_reshape = pose.dim() == 4
    if need_reshape:
        bo, t, h, w = pose.size()
        pose = pose.view(-1, h, w)
    b, h, w = pose.size()
    part = (pose / 2 + 0.5) * 24
    mask = torch.cuda.ByteTensor(b, n_parts, h, w).fill_(0)
    for i in range(n_parts):
        for j in part_groups[i]:            
            mask[:, i] = mask[:, i] | ((part > j-0.1) & (part < j+0.1)).byte()
    if need_reshape:
        mask = mask.view(bo, t, -1, h, w)
    return mask.float()

### obtain mask of faces
def get_face_mask(pose):
    if pose.dim() == 3:
        pose = pose.unsqueeze(1)            
    b, t, h, w = pose.size()
    part = (pose / 2 + 0.5) * 24
    if pose.is_cuda:
        mask = torch.cuda.ByteTensor(b, t, h, w).fill_(0)
    else:
        mask = torch.ByteTensor(b, t, h, w).fill_(0)
    for j in [23,24]:            
        mask = mask | ((part > j-0.1) & (part < j+0.1)).byte()
    return mask.float()

### remove partial labels in the pose map if necessary
def use_valid_labels(opt, pose):
    if 'pose' not in opt.dataset_mode: return pose
    if pose is None: return pose
    if type(pose) == list:
        return [use_valid_labels(opt, p) for p in pose]
    assert(pose.dim() == 4 or pose.dim() == 5)
    if opt.pose_type == 'open':
        if pose.dim() == 4: pose = pose[:,3:]
        elif pose.dim() == 5: pose = pose[:,:,3:]
    elif opt.remove_face_labels:        
        if pose.dim() == 4:
            face_mask = get_face_mask(pose[:,2])            
            pose = torch.cat([pose[:,:3] * (1 - face_mask) - face_mask, pose[:,3:]], dim=1)
        else:   
            face_mask = get_face_mask(pose[:,:,2]).unsqueeze(2)            
            pose = torch.cat([pose[:,:,:3] * (1 - face_mask) - face_mask, pose[:,:,3:]], dim=2)        
    return pose 

def remove_dummy_from_tensor(opt, tensors, remove_size=0):    
    if remove_size == 0: return tensors
    if tensors is None: return None
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(opt, tensor, remove_size) for tensor in tensors]    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors