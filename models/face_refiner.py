# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch
import torch.nn.functional as F

from .base_model import BaseModel

class FaceRefineModel(BaseModel):
    def name(self):
        return 'FaceRefineModel'

    def initialize(self, opt, add_face_D, refine_face):
        BaseModel.initialize(self, opt) 
        self.opt = opt
        self.add_face_D = add_face_D
        self.refine_face = refine_face
        self.face_size = int(opt.fineSize / opt.aspect_ratio) // 4       

    ### refine the face region of the fake image
    def refine_face_region(self, netGf, label_valid, fake_image, label, ref_label_valid, ref_image, ref_label):        
        label_face, fake_face_coarse = self.crop_face_region([label_valid, fake_image], label, crop_smaller=4)
        ref_label_face, ref_image_face = self.crop_face_region([ref_label_valid, ref_image], ref_label, crop_smaller=4)
        fake_face = netGf(label_face, ref_label_face.unsqueeze(1), ref_image_face.unsqueeze(1), img_coarse=fake_face_coarse.detach())
        fake_image = self.replace_face_region(fake_image, fake_face, label, fake_face_coarse.detach(), crop_smaller=4)
        return fake_image

    ### crop out the face region of the image (and resize if necessary to feed into generator/discriminator)
    def crop_face_region(self, image, input_label, crop_smaller=0):
        if type(image) == list:
            return [self.crop_face_region(im, input_label, crop_smaller) for im in image]        
        for i in range(input_label.size(0)):
            ys, ye, xs, xe = self.get_face_region(input_label[i:i+1], crop_smaller=crop_smaller)
            output_i = F.interpolate(image[i:i+1,-3:,ys:ye,xs:xe], size=(self.face_size, self.face_size))
            output = torch.cat([output, output_i]) if i != 0 else output_i        
        return output

    ### replace the face region in the fake image with the refined version
    def replace_face_region(self, fake_image, fake_face, input_label, fake_face_coarse=None, crop_smaller=0):
        fake_image = fake_image.clone()
        b, _, h, w = input_label.size()
        for i in range(b):
            ys, ye, xs, xe = self.get_face_region(input_label[i:i+1], crop_smaller)
            fake_face_i = fake_face[i:i+1] + (fake_face_coarse[i:i+1] if fake_face_coarse is not None else 0)
            fake_face_i = F.interpolate(fake_face_i, size=(ye-ys, xe-xs), mode='bilinear')            
            fake_image[i:i+1,:,ys:ye,xs:xe] = torch.clamp(fake_face_i, -1, 1)            
        return fake_image

    ### get coordinates of the face bounding box
    def get_face_region(self, pose, crop_smaller=0):        
        if pose.dim() == 3: pose = pose.unsqueeze(0)
        elif pose.dim() == 5: pose = pose[-1,-1:]
        _, _, h, w = pose.size()
             
        use_openpose = not self.opt.basic_point_only and not self.opt.remove_face_labels
        if use_openpose: # use openpose face keypoints to identify face region   
            face = ((pose[:,-3] > 0) & (pose[:,-2] > 0) & (pose[:,-1] > 0)).nonzero()
        else: # use densepose labels
            face = (pose[:,2] > 0.9).nonzero()
        if face.size(0):
            y, x = face[:,1], face[:,2]
            ys, ye, xs, xe = y.min().item(), y.max().item(), x.min().item(), x.max().item()                        
            if use_openpose:
                xc, yc = (xs + xe) // 2, (ys*3 + ye*2) // 5
                ylen = int((xe - xs) * 2.5)
            else:
                xc, yc = (xs + xe) // 2, (ys + ye) // 2
                ylen = int((ye - ys) * 1.25)
            ylen = xlen = min(w, max(32, ylen))
            yc = max(ylen//2, min(h-1 - ylen//2, yc))
            xc = max(xlen//2, min(w-1 - xlen//2, xc))   
        else:            
            yc = h//4
            xc = w//2
            ylen = xlen = h // 32 * 8                               
                
        ys, ye, xs, xe = yc - ylen//2, yc + ylen//2, xc - xlen//2, xc + xlen//2
        if crop_smaller != 0: # crop slightly smaller region inside face
            ys += crop_smaller; xs += crop_smaller
            ye -= crop_smaller; xe -= crop_smaller
        return ys, ye, xs, xe   