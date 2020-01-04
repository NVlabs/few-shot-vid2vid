# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
from download_gdrive import *
os.system('cd models/networks/flownet2_pytorch/; bash install.sh; cd ../../../')

file_id = '1E8re-b6csNuo-abg1vJKCDjCzlIam50F'
chpt_path = './models/networks/flownet2_pytorch/'
destination = os.path.join(chpt_path, 'FlowNet2_checkpoint.pth.tar')
download_file_from_google_drive(file_id, destination) 