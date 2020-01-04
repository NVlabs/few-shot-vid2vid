# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
from download_gdrive import *

file_id = '1NyxrzJbHgDOpf-nRJhfxsHqWNDW_D1TV'
chpt_path = './datasets/'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)
destination = os.path.join(chpt_path, 'datasets.zip')
download_file_from_google_drive(file_id, destination) 
unzip_file(destination, chpt_path)