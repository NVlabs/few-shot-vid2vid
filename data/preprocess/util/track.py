# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import glob
import os.path as path
from numpy import linalg as LA
import json

from util.util import get_keypoint_array, get_valid_openpose_keypoints
from util.check_valid import is_full_body, has_overlap


# For checking a frame is valid or not.
conf_thre = 0.01         # threshold for pose estimation confidence
min_body_len = 256       # minimum body length to use

# Tracking related params.
track_torso_only = True  # only track the torso region
pos_diff_val_thre = 100  # maximum pose difference between neighboring frames for each joint
pos_diff_num_thre = 10   # minimum number of joints whose pose differences required to be within threshold
next_conf_thre = 0.5     # ratio of best pose match to second best confidence


def track_persons(keypoint_dicts_prev, keypoint_dicts_now, ppl_indices_prev):
    ppl_indices_now = [-1] * len(ppl_indices_prev)
    ppl_indices_tmp = []
    # Go through the ppl in dict_now. If a person is valid, add it to new_indices.
    for i, keypoint_dict_now in enumerate(keypoint_dicts_now):
        pose_pts = get_keypoint_array(keypoint_dict_now)
        full_body = is_full_body(pose_pts)
        valid_pts = get_valid_openpose_keypoints(pose_pts)
        enough_valid_pts = valid_pts.shape[0] >= 5
        enough_res = (valid_pts[:, 1].max() - valid_pts[:, 1].min()) \
                     >= min_body_len
        if full_body and enough_valid_pts and enough_res:
            ppl_indices_tmp.append(i)
    keypoint_dicts_now = [keypoint_dicts_now[i] for i in ppl_indices_tmp]
    if len(keypoint_dicts_now) == 0:
        return ppl_indices_now

    # For each person in dicts_prev, try to find the corresponding person in
    # dicts_now.
    for p, ppl_idx in enumerate(ppl_indices_prev):
        if ppl_idx != -1:
            keypoint_dict_prev = keypoint_dicts_prev[ppl_idx]
            pose_pts_prev = get_keypoint_array(keypoint_dict_prev)
            cur_min = cur_second_min = 10000
            cur_i = -1

            # First, remove overlapping people in the frame, and only keep people
            # that do not overlap with others.
            all_pose_pts = [get_keypoint_array(k) for k in keypoint_dicts_now]
            valid_pose_pts = []
            for i, pose_pts in enumerate(all_pose_pts):
                overlap = False
                for j, pose_pts2 in enumerate(all_pose_pts):
                    if i == j:
                        continue
                    overlap = overlap | has_overlap(pose_pts, pose_pts2)
                    if overlap:
                        break
                if not overlap:
                    valid_pose_pts += [pose_pts]

            # Enumerate all people in dicts_now to find the person with the
            # closest pose keypoints.
            for i, pose_pts_now in enumerate(valid_pose_pts):
                pos_diff = abs(pose_pts_prev - pose_pts_now)[:, :2]
                # If a keypoint is not detected, set the difference to some
                # high value.
                not_valid = (pose_pts_prev[:, 2] < conf_thre) | (
                            pose_pts_now[:, 2] < conf_thre)
                pos_diff[not_valid, :] = 1000

                # Check if the person has smaller pose difference than
                # current min.
                if track_torso_only:
                    dist1 = LA.norm(pos_diff[1])
                    dist2 = LA.norm(pos_diff[8])
                    dist = dist1 + dist2
                    criteria = dist1 < pos_diff_val_thre and \
                               dist2 < pos_diff_val_thre and \
                               dist < cur_min
                else:
                    dist = pos_diff.sum()
                    criteria = ((pos_diff.sum(1) < pos_diff_val_thre).sum()
                                > pos_diff_num_thre) \
                               and dist < cur_min
                # If so, set the current min to this person.
                if criteria:
                    cur_second_min = cur_min
                    cur_min = dist
                    cur_i = i

            # Also check second to best ratio to remove false matching.
            if cur_i != -1 and (cur_min / cur_second_min) < next_conf_thre:
                ppl_indices_now[p] = ppl_indices_tmp[cur_i]
                ppl_indices_tmp[cur_i] = -1

    # If a person in dicts_now is not matched, assume a new person appears and
    # assign to him a new index.
    # First, find the next available index that is not used now.
    available_idx = 0
    while ppl_indices_prev[available_idx] != -1 or \
            ppl_indices_now[available_idx] != -1:
        available_idx += 1
    # Then assign the index to that person.
    for new_person_idx in ppl_indices_tmp:
        if new_person_idx != -1:
            ppl_indices_now[available_idx] = new_person_idx
            while ppl_indices_prev[available_idx] != -1 or \
                    ppl_indices_now[available_idx] != -1:
                available_idx += 1
    return ppl_indices_now


# Divide each sequence into subsequences based on motions and scene transitions.
# If multiple people exist in a frame, each person will become a subsequence.
def divide_sequences(args, video_idx):
    op_dir = path.join(args.output_root, args.openpose_folder, video_idx)
    json_paths = sorted(glob.glob(op_dir + '/*.json'))
    keypoint_dicts_prev = None
    # start and end frame indices
    recorded_start_indices, recorded_end_indices = [], []
    # indices for the particular person.
    recorded_ppl_indices = []
    n_max_ppl = 50  # max number of people in a sequence
    all_ppl_indices, ppl_indices = [], [-1] * n_max_ppl
    start_indices = [0] * n_max_ppl
    # all_ppl_indices records the indices in openpose for all people in the
    # sequence. For example, if we assume a maximum of 4 people,
    # [[ 3, -1, -1, -1],  # frame 1
    #  [ 1,  0, -1, -1],  # frame 2
    #  [-1,  2, -1, -1]]  # frame 3
    # means a person is the 3rd person (in openpose) in frame 1 and 1st person
    # in frame 2, and disappears in frame 3. Another person appears in frame 2
    # (0th in openpose) and frame 3 (2nd). -1 means not found.

    for i, json_path in enumerate(json_paths):
        with open(json_path, encoding='utf-8') as f:
            keypoint_dicts = json.loads(f.read())["people"]

        # Get the corresponding people indices in neighboring frames.
        ppl_indices = track_persons(keypoint_dicts_prev, keypoint_dicts,
                                    ppl_indices)
        all_ppl_indices.append(ppl_indices)

        ppl_indices_prev, ppl_indices_now = all_ppl_indices[i-1], \
                                            all_ppl_indices[i]
        for p in range(len(ppl_indices_prev)):
            ppl_idx_prev, ppl_idx_now = ppl_indices_prev[p], \
                                        ppl_indices_now[p]
            if ppl_idx_prev == -1 and ppl_idx_now != -1:
                # A new person appears in the current frame. Start recording
                # the frames this person appears in.
                start_indices[p] = i
            elif ppl_idx_prev != -1 and \
                    ((ppl_idx_now == -1) or (i == len(json_paths)-1)):
                # A person disappears in the current frame or it is the
                # last frame of the sequence. Get the frames that person
                # appeared, and form a subsequence using these frames.
                if ppl_idx_now != -1:
                    end_idx = i
                start_idx = start_indices[p]
                if (end_idx - start_idx) > args.min_n_of_frames:
                    this_ppl_indices = [indices[p] for indices in
                                        all_ppl_indices[start_idx:end_idx]]
                    recorded_start_indices.append(start_idx)
                    recorded_end_indices.append(end_idx)
                    recorded_ppl_indices.append(this_ppl_indices)
        keypoint_dicts_prev = keypoint_dicts
        end_idx = i

    print('Number of sub-sequences: ', len(recorded_start_indices))
    return recorded_start_indices, recorded_end_indices, recorded_ppl_indices
