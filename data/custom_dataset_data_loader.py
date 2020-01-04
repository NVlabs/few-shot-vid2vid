# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import torch.utils.data
import torch.distributed as dist
from data.base_data_loader import BaseDataLoader
import data

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = data.create_dataset(opt)
        if dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            sampler = None

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=(sampler is None) and not opt.serial_batches,
            sampler=sampler,
            pin_memory=True,
            num_workers=int(opt.nThreads),
            drop_last=True
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        size = min(len(self.dataset), self.opt.max_dataset_size)
        ngpus = len(self.opt.gpu_ids)
        round_to_ngpus = (size // ngpus) * ngpus
        return round_to_ngpus
