### Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
### Licensed under the Nvidia Source Code License.
import os
import numpy as np
import torch
import cv2
from collections import OrderedDict

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

### setup dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

### setup models
model = create_model(opt)
model.eval()
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
if opt.finetune: web_dir += '_finetune'
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch), infer=True)

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many or i >= len(dataset): break
    img_path = data['path']   
    data_list = [data['tgt_label'], data['tgt_image'], None, None, data['ref_label'], data['ref_image'], None, None]
    synthesized_image, _, _, _, _, _ = model(data_list)
            
    synthesized_image = util.tensor2im(synthesized_image)    
    tgt_image = util.tensor2im(data['tgt_image'])    
    ref_image = util.tensor2im(data['ref_image'], tile=True)    
    seq = data['seq'][0]
    visual_list = [ref_image, tgt_image, synthesized_image]        
    visuals = OrderedDict([(seq, np.hstack(visual_list)),    					   
                           (seq + '/synthesized', synthesized_image),
                           (seq + '/ref_image', ref_image if i == 0 else None),
                          ])
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)
