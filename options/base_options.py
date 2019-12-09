### Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
### Licensed under the Nvidia Source Code License.
import os
import sys
import argparse
import pickle
import torch

import data
from util import util
from util.distributed import master_only_print as print

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):    
        # experiment specifics
        parser.add_argument('--name', type=str, default='test', help='name of the experiment. It decides where to store samples and models')        
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')                       
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='vid2vid', help='which model to use')                
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')

        # input/output sizes       
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--batch_for_first_gpu', type=int, default=-1, help='input batch size for first GPU. if -1, same as the other GPUs')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')        
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # for setting inputs
        parser.add_argument('--dataroot', type=str, default='datasets/pose/')
        parser.add_argument('--dataset_mode', type=str, default='fewshot_pose')        
        parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')        
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation') 
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')                
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

         # for displays
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        parser.add_argument('--use_visdom', action='store_true', help='if specified, use Visdom. Requires visdom installed')
        parser.add_argument('--visdom_id', type=int, default=0)

    
        # for generator        
        parser.add_argument('--netG', type=str, default='fewshot', help='selects model to use for netG')
        parser.add_argument('--n_downsample_G', type=int, default=5, help='# of downsamplings in netG')        
        parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch', help='instance normalization or batch normalization')
        parser.add_argument('--ks', type=int, default=1, help='filter size for convolution in SPADE')
        parser.add_argument('--netS', type=str, default='encoderdecoder', help='selects model to use for the label embedding network')                
        
        # for reference image encoder        
        parser.add_argument('--use_label_ref', type=str, default='mul', help='how to use the reference label: concat | mul')
        parser.add_argument('--res_for_ref', action='store_true', help='use residual blocks instead of convs in reference image encoder')
        
        # for adaptive weight generation
        parser.add_argument('--adaptive_conv', action='store_true', help='if specified, use adaptive convolution for main branch in generator')
        parser.add_argument('--adaptive_spade', action='store_true', help='if specified, use adaptive convolution in the SPADE module')
        parser.add_argument('--no_adaptive_embed', action='store_true', help='if specified, do not use adaptive convolution in the embedding network in SPADE')
        parser.add_argument('--n_adaptive_layers', type=int, default=4, help='# of adaptive layers')
        parser.add_argument('--n_fc_layers', type=int, default=2, help='# of fc layers for weight generation')

        # for temporal and flow generation
        parser.add_argument('--n_frames_G', type=int, default=2, help='number of input frames to feed into generator, i.e., n_frames_G-1 is the number of frames we look into past')
        parser.add_argument('--n_frames_per_gpu', type=int, default=1, help='the number of frames to load into one GPU at a time. only 1 is supported now')
        parser.add_argument('--no_flow_gt', action='store_true', help='do not compute ground truth flow for training loss')
        parser.add_argument('--n_downsample_F', type=int, default=3, help='number of downsamplings in flow network')
        parser.add_argument('--nff', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--n_blocks_F', type=int, default=6, help='number of residual blocks in flow network')
        parser.add_argument('--norm_F', type=str, default='spectralsyncbatch', help='instance normalization or batch normalization')
        parser.add_argument('--flow_multiplier', type=int, default=20, help='flow output multiplier')
        parser.add_argument('--flow_deconv', action='store_true', help='use deconvolution for flow generation')
        parser.add_argument('--spade_combine', action='store_true', help='use SPADE to combine with warped image instead of linear combination')
        parser.add_argument('--n_sc_layers', type=int, default=3, help='number of layers to use SPADE combination in netG')

        # for attention mechanism
        parser.add_argument('--n_shot', type=int, default=1, help='how many reference images')
        parser.add_argument('--n_downsample_A', type=int, default=2, help='# of downsamplings in attention network')
        parser.add_argument('--warp_ref', action='store_true', help='if specified, warp the reference image and combine with the synthesized image')

        # for discriminators        
        parser.add_argument('--which_model_netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--netD_subarch', type=str, default='n_layers', help='(n_layers|resnet_{}down{}blocks)')
        parser.add_argument('--num_D', type=int, default=1, help='number of discriminators to use')
        parser.add_argument('--n_layers_D', type=int, default=4, help='only used if which_model_netD==n_layers')
        parser.add_argument('--ndf', type=int, default=32, help='# of discrim filters in first conv layer')    
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')        
        parser.add_argument('--add_face_D', action='store_true', help='add additional discriminator for face region')
        parser.add_argument('--adaptive_D_layers', type=int, default=1, help='number of adaptive layers in discriminator')

        parser.add_argument('--lambda_kld', type=float, default=0.0, help='weight for KLD loss')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_temp', type=float, default=2.0, help='weight for temporal loss')
        parser.add_argument('--lambda_flow', type=float, default=10.0, help='weight for flow')        
        parser.add_argument('--lambda_weight', type=float, default=10.0, help='weight for flow mask')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')        
        parser.add_argument('--lambda_face', type=float, default=10.0, help='weight for face region')        

        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')

        # for optimizer
        parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        
        parser.add_argument('--finetune', action='store_true', help='finetune the generator during inference based on the given reference')
        parser.add_argument('--fp16', action='store_true')
        parser.add_argument('--distributed', action='store_true', help='distributed training')
        parser.add_argument('--local_rank', type=int, default=0)
        
        self.initialized = True
        return parser

    def gather_options(self):

        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)            
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()        

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()                
        opt = parser.parse_args()

        self.parser = parser
        
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)
        

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

                
    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        #self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)
        

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        if opt.isTrain and opt.debug:
            opt.name = 'test'
            opt.ngf = opt.ndf = 4            
            opt.display_freq = opt.print_freq = opt.niter = opt.niter_decay = opt.niter_step = opt.niter_single = 1        
            opt.max_dataset_size = opt.batchSize * 8
            opt.save_latest_freq = 100

        self.opt = opt
        return self.opt
