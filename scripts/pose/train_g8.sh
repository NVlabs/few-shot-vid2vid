# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt

python train.py --name pose --dataset_mode fewshot_pose \
--adaptive_spade --warp_ref --spade_combine --remove_face_labels --add_face_D \
--niter_single 100 --niter 200 \
--gpu_ids 0,1,2,3,4,5,6,7 --batchSize 30 --nThreads 16 --continue_train 