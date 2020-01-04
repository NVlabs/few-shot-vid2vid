# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import lmdb
import pickle
import numpy as np
from PIL import Image
import cv2
import torch.utils.data as data
from util.distributed import master_only_print as print

class LMDBDataset(data.Dataset):
    def __init__(self, root, write_cache=False):
        self.root = os.path.expanduser(root)
        self.env = lmdb.open(root, max_readers=126, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        print('LMDB file at %s opened.' % root)
        cache_file = os.path.join(root, '_cache_')
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        elif write_cache:
            print('generating keys')
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))
            print('cache file generated at %s' % cache_file)
        else:
            self.keys = []

    def getitem_by_path(self, path, is_img=True):        
        env = self.env
        with env.begin(write=False) as txn:
            buf = txn.get(path)        
        if is_img:
            img = cv2.imdecode(np.fromstring(buf, dtype=np.uint8), 1)
            img = Image.fromarray(img)
            return img, path
        return buf, path

    def __getitem__(self, index):
        path = self.keys[index]
        return self.getitem_by_path(path)

    def __len__(self):
        return self.length
