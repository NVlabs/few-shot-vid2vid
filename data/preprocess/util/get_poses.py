# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
# To view a copy of this license, visit
# https://nvlabs.github.io/few-shot-vid2vid/License.txt
import os
import glob
import os.path as path
import cv2

from util.util import makedirs, remove_frame
from util.check_valid import remove_invalid_frames, remove_static_frames, \
    remove_isolated_frames, is_valid_frame


# Run OpenPose on the extracted frames, and remove invalid frames.
# To expedite the process, we will first process only keyframes in the video.
# If the keyframe looks promising, we will then process the whole block of
# frames for the keyframe.
def run_openpose(args, video_path):
    video_idx = path.basename(video_path).split('.')[0]
    try:
        img_dir = path.join(args.output_root, args.img_folder, video_idx)
        op_dir = path.join(args.output_root, args.openpose_folder, video_idx)
        img_names = sorted(glob.glob(img_dir + '/*.jpg'))
        op_names = sorted(glob.glob(op_dir + '/*.json'))

        # If the frames haven't been extracted or OpenPose hasn't been run or
        # finished processing.
        if (not os.path.isdir(img_dir) or not os.path.isdir(op_dir)
                or len(img_names) != len(op_names)):
            makedirs(img_dir)

            # First run OpenPose on key frames, then decide whether to run
            # the whole batch of frames.
            extract_key_frames(args, video_path, img_dir)
            run_openpose_cmd(args, video_idx)

            # If key frame looks good, extract all frames in the batch and
            # run OpenPose.
            if args.n_skip_frames > 1:
                extract_valid_frames(args, video_path, img_dir)
                run_openpose_cmd(args, video_idx)

            # Remove all unusable frames.
            remove_invalid_frames(args, video_idx)
            remove_static_frames(args, video_idx)
            remove_isolated_frames(args, video_idx)
    except:
        raise ValueError('video %s running openpose failed' % video_idx)


# Run DensePose on the extracted frames, and remove invalid frames.
def run_densepose(args, video_idx):
    try:
        img_dir = path.join(args.output_root, args.img_folder, video_idx)
        dp_dir = path.join(args.output_root, args.densepose_folder, video_idx)
        img_names = sorted(glob.glob(img_dir + '/*.jpg'))
        dp_names = sorted(glob.glob(dp_dir + '/*.png'))

        if not os.path.isdir(dp_dir) or len(img_names) != len(dp_names):
            makedirs(dp_dir)

            # Run densepose.
            run_densepose_cmd(args, video_idx)
    except:
        raise ValueError('video %s running densepose failed' % video_idx)


# Extract only the keyframes in the video.
def extract_key_frames(args, video_path, img_dir):
    print('Extracting keyframes.')
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0
    while success:
        if frame_count % args.n_skip_frames == 0:
            write_name = path.join(img_dir, "frame%06d.jpg" % frame_count)
            cv2.imwrite(write_name, image)
        success, image = vidcap.read()
        frame_count += 1


# Extract valid frames from the video based on the extracted keyframes.
def extract_valid_frames(args, video_path, img_dir):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0
    do_write = True
    while success:
        is_key_frame = frame_count % args.n_skip_frames == 0
        write_name = path.join(img_dir, "frame%06d.jpg" % frame_count)
        # Each time it's keyframe, check whether the frame is valid. If it is,
        # all frames following it before the next keyframe will be extracted.
        # Otherwise, this block of frames will be skipped and the next keyframe
        # will be examined.
        if is_key_frame:
            do_write = is_valid_frame(args, write_name)
            if not do_write:  # If not valid, remove this keyframe.
                remove_frame(args, start=write_name)
        if do_write:
            cv2.imwrite(write_name, image)
        success, image = vidcap.read()
        frame_count += 1
    print('Video contains %d frames.' % frame_count)


# Extract all frames from the video.
def extract_all_frames(args, video_path):
    video_idx = path.basename(video_path).split('.')[0]
    img_dir = path.join(args.output_root, args.img_folder, video_idx)
    makedirs(img_dir)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0
    while success:
        write_name = path.join(img_dir, "frame%06d.jpg" % frame_count)
        cv2.imwrite(write_name, image)
        success, image = vidcap.read()
        frame_count += 1
    print('Extracted %d frames' % frame_count)


# Running the actual OpenPose command.
def run_openpose_cmd(args, video_idx):
    pwd = os.getcwd()
    img_dir = path.join(pwd, args.output_root, args.img_folder, video_idx)
    op_dir = path.join(pwd, args.output_root, args.openpose_folder, video_idx)
    render_dir = path.join(pwd, args.output_root,
                           args.openpose_folder + '_rendered', video_idx)
    makedirs(op_dir)
    makedirs(render_dir)

    cmd = 'cd %s; ./build/examples/openpose/openpose.bin --display 0 ' \
          '--disable_blending --image_dir %s --write_images %s --face --hand ' \
          '--face_render_threshold 0.1 --hand_render_threshold 0.02 ' \
          '--write_json %s; cd %s' \
          % (args.openpose_root, img_dir, render_dir, op_dir,
             path.join(pwd, args.output_root))
    os.system(cmd)


# Running the actual DensePose command.
def run_densepose_cmd(args, video_idx):
    pwd = os.getcwd()
    img_dir = path.join(pwd, args.output_root, args.img_folder, video_idx)
    dp_dir = path.join(pwd, args.output_root, args.densepose_folder, video_idx, 'frame.png')
    cmd = 'python2 tools/infer_simple.py ' \
          '--cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml ' \
          '--wts https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl ' \
          '--output-dir %s %s' % (dp_dir, img_dir)
    # cmd = 'python apply_net.py show configs/densepose_rcnn_R_101_FPN_s1x.yaml ' \
    #       'densepose_rcnn_R_101_FPN_s1x.pkl %s dp_segm,dp_u,dp_v --output %s' \
    #       % (img_dir, dp_dir)
    cmd = 'cd %s; ' % args.densepose_root \
          + cmd \
          + '; cd %s' % path.join(pwd, args.output_root)
    os.system(cmd)
