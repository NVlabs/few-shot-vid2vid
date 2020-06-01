# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import os.path as path
import numpy as np


# Extract keypoint array given the openpose dict.
def get_keypoint_array(keypoint_dict):
    if type(keypoint_dict) == list:
        return [get_keypoint_array(d) for d in keypoint_dict]
    if type(keypoint_dict) != np.ndarray:
        keypoint_dict = np.array(keypoint_dict["pose_keypoints_2d"]).reshape(25, 3)
    return keypoint_dict


# Only extract openpose keypoints where the confidence is larger then conf_thre.
def get_valid_openpose_keypoints(keypoint_array):
    if type(keypoint_array) == list:
        return [get_valid_openpose_keypoints(k) for k in keypoint_array]
    return keypoint_array[keypoint_array[:, 2] > 0.01, :]


# Remove particular frame(s) for all folders.
def remove_frame(args, video_idx='', start=0, end=None):
    if not isinstance(start, int):
        video_idx = path.basename(path.dirname(start))
        start = get_frame_idx(start)
    if end is None:
        end = start
    for i in range(start, end + 1):
        img_path = path.join(args.output_root, args.img_folder, video_idx,
                             'frame%06d.jpg' % i)
        op_path = path.join(args.output_root, args.openpose_folder, video_idx,
                            'frame%06d%s' % (i, args.openpose_postfix))
        dp_path = path.join(args.output_root, args.densepose_folder, video_idx,
                            'frame%06d%s' % (i, args.densepose_postfix))
        dm_path = path.join(args.output_root, args.densemask_folder, video_idx,
                            'frame%06d%s' % (i, args.densemask_postfix))
        print('removing %s' % img_path)
        remove(img_path)
        remove(op_path)
        remove(dp_path)
        remove(dm_path)


def remove_folder(args, video_idx):
    os.rmdir(path.join(args.output_root, args.img_folder, video_idx))
    os.rmdir(path.join(args.output_root, args.openpose_folder, video_idx))
    os.rmdir(path.join(args.output_root, args.densepose_folder, video_idx))
    os.rmdir(path.join(args.output_root, args.densemask_folder, video_idx))


def get_frame_idx(file_name):
    return int(path.basename(file_name)[5:11])


def makedirs(folder):
    if not path.exists(folder):
        os.umask(0)
        os.makedirs(folder, mode=0o777)


def remove(file_name):
    if path.exists(file_name):
        os.remove(file_name)
