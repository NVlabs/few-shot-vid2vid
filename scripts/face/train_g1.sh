# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

python train.py --name face --dataset_mode fewshot_face \
--adaptive_spade --warp_ref --spade_combine --batchSize 8 --continue_train