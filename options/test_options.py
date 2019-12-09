### Copyright (C) 2019 NVIDIA Corporation. All rights reserved. 
### Licensed under the Nvidia Source Code License.
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)        
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')        
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')        
        parser.add_argument('--how_many', type=int, default=300, help='how many test images to run')        
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(batchSize=1)
        parser.set_defaults(nThreads=1)
        parser.set_defaults(no_flip=True)
        self.isTrain = False
        return parser
