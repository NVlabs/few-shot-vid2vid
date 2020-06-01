# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import glob
import os.path as path
import json

from util.util import remove_frame, get_keypoint_array, get_frame_idx, \
    get_valid_openpose_keypoints


# Remove invalid frames in the video.
def remove_invalid_frames(args, video_idx):
    op_dir = path.join(args.output_root, args.openpose_folder, video_idx)
    json_paths = sorted(glob.glob(op_dir + '/*.json'))
    for json_path in json_paths:
        if not is_valid_frame(args, json_path):
            remove_frame(args, start=json_path)


# Remove static frames in the video if no motion is detected more than
# max_static_frames.
def remove_static_frames(args, video_idx):
    max_static_frames = 5  # maximum number of frames to be static
    op_dir = path.join(args.output_root, args.openpose_folder, video_idx)
    json_paths = sorted(glob.glob(op_dir + '/*.json'))
    start_idx = end_idx = 0
    keypoint_dicts_prev = None

    for json_path in json_paths:
        with open(json_path, encoding='utf-8') as f:
            keypoint_dicts = json.loads(f.read())["people"]
        is_moving = detect_motion(keypoint_dicts_prev, keypoint_dicts)
        keypoint_dicts_prev = keypoint_dicts

        i = get_frame_idx(json_path)
        if not is_moving:
            end_idx = i
        else:
            # If static frames longer than max_static_frames, remove them.
            if (end_idx - start_idx) > max_static_frames:
                remove_frame(args, video_idx, start_idx, end_idx)
            start_idx = end_idx = i


# Remove small batch frames if number of consecutive frames is smaller than
# min_n_of_frames.
def remove_isolated_frames(args, video_idx):
    op_dir = path.join(args.output_root, args.openpose_folder, video_idx)
    json_paths = sorted(glob.glob(op_dir + '/*.json'))

    if len(json_paths):
        start_idx = end_idx = get_frame_idx(json_paths[0]) - 1
        for json_path in json_paths:
            i = get_frame_idx(json_path)
            # If the frames are not consecutive, there's a breakpoint.
            if i != end_idx + 1:
                # Check if this block of frames is longer than min_n_of_frames.
                if (end_idx - start_idx) < args.min_n_of_frames:
                    remove_frame(args, video_idx, start_idx, end_idx)
                start_idx = i
            end_idx = i
        # Need to check again at the end of sequence.
        if (end_idx - start_idx) < args.min_n_of_frames:
            remove_frame(args, video_idx, start_idx, end_idx)


# Detect if motion exists between consecutive frames.
def detect_motion(keypoint_dicts_1, keypoint_dicts_2):
    motion_thre = 5  # minimum position difference to count as motion
    # If it's the first frame or the number of people in these two frames
    # are different, return true.
    if keypoint_dicts_1 is None:
        return True
    if len(keypoint_dicts_1) != len(keypoint_dicts_2):
        return True

    # If the pose difference between two frames are larger than threshold,
    # return true.
    for keypoint_dict_1, keypoint_dict_2 in zip(keypoint_dicts_1, keypoint_dicts_2):
        pose_pts1, pose_pts2 = get_keypoint_array([keypoint_dict_1, keypoint_dict_2])
        if ((abs(pose_pts1 - pose_pts2) > motion_thre) &
                (pose_pts1 != 0) & (pose_pts2 != 0)).any():
            return True
    return False


# If densepose did not find any person in the frame (and thus outputs nothing),
# remove the frame from the dataset.
def check_densepose_exists(args, video_idx):
    op_dir = path.join(args.output_root, args.openpose_folder, video_idx)
    json_paths = sorted(glob.glob(op_dir + '/*.json'))
    for json_path in json_paths:
        dp_path = json_path.replace(args.openpose_folder, args.densepose_folder)
        dp_path = dp_path.replace(args.openpose_postfix, args.densepose_postfix)
        if not os.path.exists(dp_path):
            remove_frame(args, start=json_path)


# Check if the frame is valid to use.
def is_valid_frame(args, img_path):
    if img_path.endswith('.jpg'):
        img_path = img_path.replace(args.img_folder, args.openpose_folder)
        img_path = img_path.replace('.jpg', args.openpose_postfix)
    with open(img_path, encoding='utf-8') as f:
        keypoint_dicts = json.loads(f.read())["people"]
    return len(keypoint_dicts) > 0 and is_full_body(keypoint_dicts) and \
        contains_non_overlapping_people(keypoint_dicts)


# Check if the image contains a full body.
def is_full_body(keypoint_dicts):
    if type(keypoint_dicts) != list:
        keypoint_dicts = [keypoint_dicts]
    for keypoint_dict in keypoint_dicts:
        pose_pts = get_keypoint_array(keypoint_dict)
        # Contains at least one joint of head and one joint of foot.
        full = pose_pts[[0, 15, 16, 17, 18], :].any() \
            and pose_pts[[11, 14, 19, 20, 21, 22, 23, 24], :].any()
        if full:
            return True
    return False


# Check whether two people overlap with each other.
def has_overlap(pose_pts_1, pose_pts_2):
    pose_pts_1 = get_valid_openpose_keypoints(pose_pts_1)[:, 0]
    pose_pts_2 = get_valid_openpose_keypoints(pose_pts_2)[:, 0]
    # Get the x_axis bbox of the person.
    x1_start, x1_end = int(pose_pts_1.min()), int(pose_pts_1.max())
    x2_start, x2_end = int(pose_pts_2.min()), int(pose_pts_2.max())
    if x1_end < x2_start or x2_end < x1_start:
        return False
    return True


# Check if the image contains at least one person that does not overlap with others.
def contains_non_overlapping_people(keypoint_dicts):
    if len(keypoint_dicts) < 2:
        return True

    all_pose_pts = [get_keypoint_array(k) for k in keypoint_dicts]
    for i, pose_pts in enumerate(all_pose_pts):
        overlap = False
        for j, pose_pts2 in enumerate(all_pose_pts):
            if i == j:
                continue
            overlap = overlap | has_overlap(pose_pts, pose_pts2)
            if overlap:
                break
        if not overlap:
            return True
    return False
