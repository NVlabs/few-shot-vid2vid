# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import glob
import os.path as path
import argparse
import json
from tqdm import tqdm

from util.get_poses import extract_all_frames, run_densepose, run_openpose
from util.check_valid import remove_invalid_frames, remove_static_frames, \
    remove_isolated_frames, check_densepose_exists
from util.track import divide_sequences
from util.util import remove_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', default='all',
                        help='all | extract_frames | openpose | densepose | clean | divide_sequences')
    parser.add_argument('--video_root', required=True,
                        help='path for videos to process')
    parser.add_argument('--output_root', required=True,
                        help='path for output images')

    parser.add_argument('--img_folder', default='images')
    parser.add_argument('--openpose_folder', default='openpose')
    parser.add_argument('--openpose_postfix', default='_keypoints.json')
    parser.add_argument('--densepose_folder', default='densepose')
    parser.add_argument('--densepose_postfix', default='_IUV.png')
    parser.add_argument('--densemask_folder', default='densemask')
    parser.add_argument('--densemask_postfix', default='_INDS.png')
    parser.add_argument('--track_folder', default='tracking')

    parser.add_argument('--openpose_root', default='/',
                        help='root for the OpenPose library')
    parser.add_argument('--densepose_root', default='/',
                        help='root for the DensePose library')

    parser.add_argument('--n_skip_frames', type=int, default='100',
                        help='Number of frames between keyframes. A larger '
                             'number can expedite processing but may lose data')
    parser.add_argument('--min_n_of_frames', type=int, default='30',
                        help='Minimum number of frames in the output sequence.')

    args = parser.parse_args()
    return args


def rename_videos(video_root):
    video_paths = sorted(glob.glob(video_root + '/*.mp4'))
    for i, video_path in enumerate(video_paths):
        new_path = video_root + ('/%05d.mp4' % i)
        os.rename(video_path, new_path)


# Remove frames that are not suitable for training.
def remove_unusable_frames(args, video_idx):
    remove_invalid_frames(args, video_idx)
    check_densepose_exists(args, video_idx)
    remove_static_frames(args, video_idx)
    remove_isolated_frames(args, video_idx)
    video_path = path.join(args.output_root, args.img_folder, video_idx)
    if len(os.listdir(video_path)) == 0:
        remove_folder(args, video_idx)


if __name__ == "__main__":
    args = parse_args()
    if args.steps == 'all':
        args.steps = 'openpose,densepose,clean,divide_sequences'

    if 'extract_frames' in args.steps or 'openpose' in args.steps or \
            'densepose' in args.steps:
        rename_videos(args.video_root)
        video_paths = sorted(glob.glob(args.video_root + '/*.mp4'))
        for video_path in tqdm(video_paths):
            if 'extract_frames' in args.steps:
                # Only extract frames from all the videos.
                extract_all_frames(args, video_path)
            if 'openpose' in args.steps:
                # Include extracting frames and running OpenPose.
                run_openpose(args, video_path)
            if 'densepose' in args.steps:
                # Run DensePose.
                video_idx = path.basename(video_path).split('.')[0]
                run_densepose(args, video_idx)

    if 'clean' in args.steps:
        # Frames already extracted and openpose / densepose already run, only
        # remove the unusable frames in the dataset.
        # Note that the folder structure should be
        # [output_root] / [img_folder] / [sequences] / [frames], and the
        # names of the frames must be in format of
        # 'frame000000.jpg', 'frame000001.jpg', ...
        video_indices = sorted(glob.glob(path.join(args.output_root,
                                                   args.img_folder, '*')))
        video_indices = [path.basename(p) for p in video_indices]

        # Remove all unusable frames in the sequences.
        for i, video_idx in enumerate(tqdm(video_indices)):
            remove_unusable_frames(args, video_idx)

    if 'divide_sequences' in args.steps:
        # Finally, divide the remaining sequences into sub-sequences, where
        # each seb-sequence only contains one person.
        video_indices = sorted(glob.glob(path.join(args.output_root,
                                                   args.img_folder, '*')))
        video_indices = [path.basename(p) for p in video_indices]
        seq_indices = []
        start_frame_indices, end_frame_indices, ppl_indices = [], [], []
        for i, video_idx in enumerate(tqdm(video_indices)):
            start_frame_indices_i, end_frame_indices_i, ppl_indices_i = \
                divide_sequences(args, video_idx)
            seq_indices += [i] * len(start_frame_indices_i)
            start_frame_indices += start_frame_indices_i
            end_frame_indices += end_frame_indices_i
            ppl_indices += ppl_indices_i

        output = dict()
        output['seq_indices'] = seq_indices
        output['start_frame_indices'] = start_frame_indices
        output['end_frame_indices'] = end_frame_indices
        output['ppl_indices'] = ppl_indices
        output_path = path.join(args.output_root, 'all_subsequences.json')
        with open(output_path, 'w') as fp:
            json.dump(output, fp, indent=4)
