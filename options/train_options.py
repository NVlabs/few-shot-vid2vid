# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')        
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')        
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--print_mem', action='store_true', help='print memory usage')
        parser.add_argument('--print_G', action='store_true', help='print network G')
        parser.add_argument('--print_D', action='store_true', help='print network D')

        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')        
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')        
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')    
        parser.add_argument('--niter_single', type=int, default=50, help='# of iter for single frame training')
        parser.add_argument('--niter_step', type=int, default=10, help='# of iter to double the length of training sequence')           

        # for temporal
        parser.add_argument('--n_frames_D', type=int, default=2, help='number of frames to feed into temporal discriminator')
        parser.add_argument('--n_frames_total', type=int, default=2, help='the overall number of frames in a sequence to train with')
        parser.add_argument('--max_t_step', type=int, default=4, help='max spacing between neighboring sampled frames. If greater than 1, the network may randomly skip frames during training.')

        self.isTrain = True
        return parser
