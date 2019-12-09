import re
import torch
import numpy as np
from PIL import Image
import os
import cv2

from models.input_process import use_valid_labels

def visualize_label(opt, label_tensor, model=None): 
    if 'pose' in opt.dataset_mode:
        if opt.add_face_D and model is not None:            
            ys, ye, xs, xe = model.module.faceRefiner.get_face_region(label_tensor)            
        label_tensor = use_valid_labels(opt, label_tensor)

    if label_tensor.dim() == 5:
        label_tensor = label_tensor[-1]
    if label_tensor.dim() == 4:        
        label_tensor = label_tensor[-1]
    if opt.label_nc:
        visual_label = tensor2label(label_tensor[:opt.label_nc], opt.label_nc)
    else:
        visual_label = tensor2im(label_tensor[:3] if label_tensor.shape[0] >= 3 else label_tensor[:1])

    if 'pose' in opt.dataset_mode:        
        image2 = tensor2im(label_tensor[-3:])
        visual_label[image2 != 0] = image2[image2 != 0]
        if opt.add_face_D and model is not None and ys is not None:
            visual_label[ys, xs:xe, :] = visual_label[ye-1, xs:xe, :] \
                = visual_label[ys:ye, xs, :] = visual_label[ys:ye, xe-1, :] = 255
            
    if len(visual_label.shape) == 2: visual_label = np.repeat(visual_label[:,:,np.newaxis], 3, axis=2)        
    return visual_label

# Converts a Tensor into a Numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if image_tensor is None: return None
    if isinstance(image_tensor, list):
        image_tensor = [t for t in image_tensor if t is not None]
        if not image_tensor: return None
        images_np = [tensor2im(t, imtype, normalize) for t in image_tensor]
        return tile_images(images_np) if tile else images_np
    
    if image_tensor.dim() == 5:
        image_tensor = image_tensor[-1]
    if image_tensor.dim() == 4:
        if tile:            
            images_np = [tensor2im(image_tensor[b]) for b in range(image_tensor.size(0))]
            return tile_images(images_np)
        image_tensor = image_tensor[-1]
    elif image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)

    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        #image_numpy = image_numpy[:,:,0]
        image_numpy = np.repeat(image_numpy, 3, axis=2)
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):    
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result

def tensor2flow(tensor, imtype=np.uint8, tile=False):
    if tensor is None: return None
    if isinstance(tensor, list):
        tensor = [t for t in tensor if t is not None]
        if not tensor: return None
        images_np = [tensor2flow(t, imtype) for t in tensor]        
        return tile_images(images_np) if tile else images_np        
    if tensor.dim() == 5:
        tensor = tensor[-1]
    if tensor.dim() == 4:
        if tile:
            images_np = [tensor2flow(tensor[b]) for b in range(tensor.size(0))]
            return tile_images(images_np)        
        tensor = tensor[-1]
    tensor = tensor.detach().cpu().float().numpy()
    tensor = np.transpose(tensor, (1, 2, 0))

    hsv = np.zeros((tensor.shape[0], tensor.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(tensor[..., 0], tensor[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def tile_images(imgs, picturesPerRow=2):
    """ Convert to a true list of 16x16 images"""
    if type(imgs) == list:
        if len(imgs) == 1: return imgs[0]
        imgs = [img[np.newaxis,:] for img in imgs]
        imgs = np.concatenate(imgs, axis=0)

    # Calculate how many columns
    #picturesPerColumn = imgs.shape[0]/picturesPerRow + 1*((imgs.shape[0]%picturesPerRow)!=0)
    #picturesPerColumn = int(picturesPerColumn)
    
    # Padding
    #rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow        
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)    

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):        
        tiled.append(np.concatenate([imgs[j] for j in range(i, i+picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled
    
def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape train
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                    dtype=np.uint8)
    elif N == 20: # GTA/cityscape eval
        cmap = np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30), 
                         (220,220,  0), (107,142, 35), (152,251,152), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70), 
                         (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), ( 70,130,180), (  0,  0,  0)], 
                         dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1 # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
