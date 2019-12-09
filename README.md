<img src='imgs/illustration.gif' align="right" width=200>

<br><br><br><br>

# Few-shot vid2vid
### [Project](https://nvlabs.github.io/few-shot-vid2vid/) | [YouTube](https://youtu.be/8AZBuyEuDqc) | [arXiv](https://arxiv.org/abs/1910.12713)

Pytorch implementation for few-shot photorealistic video-to-video translation. It can be used for generating human motions from poses, synthesizing people talking from edge maps, or turning semantic label maps into photo-realistic videos. The core of video-to-video translation is image-to-image translation. Some of our work in that space can be found in [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [SPADE](https://github.com/NVlabs/SPADE). <br><br>
[Few-shot Video-to-Video Synthesis](https://nvlabs.github.io/few-shot-vid2vid/)  
 [Ting-Chun Wang](https://tcwang0509.github.io/), [Ming-Yu Liu](http://mingyuliu.net/), Andrew Tao, [Guilin Liu](https://liuguilin1225.github.io/), [Jan Kautz](http://jankautz.com/), [Bryan Catanzaro](http://catanzaro.name/)  
 NVIDIA Corporation  
 In Neural Information Processing Systems (**NeurIPS**) 2019  

## Example Results
- Dance Videos
<p align='center'>
  <img src='imgs/dance.gif' width='400'/>
  <img src='imgs/statue.gif' width='400'/>
</p>

- Talking Head Videos
<p align='center'>
  <img src='imgs/face.gif' width='400'/>
  <img src='imgs/mona_lisa.gif' width='400'/>
</p>

- Street View Videos
<p align='center'>
  <img src='imgs/street.gif' width='400'/>  
</p>


## Prerequisites
- Linux or macOS
- Python 3
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.2


## Getting Started
### Installation
- Install python libraries [dominate](https://github.com/Knio/dominate) and requests.
```bash
pip install dominate requests
```
- If you plan to train with face datasets, please install dlib.
```bash
pip install dlib
```
- If you plan to train with pose datasets, please install [DensePose](https://github.com/facebookresearch/DensePose) and/or [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
- Clone this repo:
```bash
git clone https://github.com/NVlabs/few-shot-vid2vid
cd few-shot-vid2vid
```


### Dataset
- Pose
  - We use random dancing videos found on YouTube. We then apply DensePose / OpenPose to estimate the poses for each frame.
- Face
  - We use the [FaceForensics](http://niessnerlab.org/projects/roessler2018faceforensics.html) dataset. We then use landmark detection to estimate the face keypoints, and interpolate them to get face edges.
- Street
  - We use a mix of sequences from different cities, which include Cityscapes [official website](https://www.cityscapes-dataset.com/) and other cities found on YouTube.
  - We apply a pre-trained segmentation algorithm to get the corresponding semantic maps.
- Please add the obtained images to the `datasets` folder in the same way the example images are provided.

## Training
- First, compile a snapshot of [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) by running `python scripts/download_flownet2.py`.
- Please first download example datasets by running `python scripts/download_datasets.py`.
- The following scripts are examples of using one GPU. For multi-GPU training, simply increase the batch sizes.

### Training with pose datasets
- Example DensePose and OpenPose results are included. If you plan to use your own dataset, please generate these results and put them in the same way the example dataset is provided.
- Run the example script (`bash ./scripts/pose/train_g1.sh`)
  ```bash
  python train.py --name pose --dataset_mode fewshot_pose --adaptive_spade --warp_ref --spade_combine --remove_face_labels --add_face_D --niter_single 100 --niter 200 --batchSize 2
  ```
- Please refer to [More Training/Test Details](https://github.com/NVlabs/few-shot-vid2vid#more-trainingtest-details) for more explanations about training flags.

### Training with face datasets
- Run the example script (`bash ./scripts/face/train_g1.sh`)
  ```bash
  python train.py --name face --dataset_mode fewshot_face --adaptive_spade --warp_ref --spade_combine --batchSize 8
  ```

### Training with street dataset
  - Run the example script (`bash ./scripts/street/train_g1.sh`)
  ```bash
  python train.py --name street --dataset_mode fewshot_street --adaptive_spade --loadSize 512 --fineSize 512 --batchSize 6
  ```


### Training with your own dataset
- If your input is a label map, please generate label maps which are one-channel whose pixel values correspond to the object labels (i.e. 0,1,...,N-1, where N is the number of labels). This is because we need to generate one-hot vectors from the label maps. Please use `--label_nc N` during both training and testing.
- If your input is not a label map, please specify `--input_nc N` where N is the number of input channels (The default is 3 for RGB images).
- The default setting for preprocessing is `scale_width`, which will scale the width of all training images to `opt.loadSize` while keeping the aspect ratio. If you want a different setting, please change it by using the `--resize_or_crop` option.



## Testing 
- After training, you can run inference by using the following scripts. The test results will be saved in: `./results/`. Due to privacy concerns, the pretrained models are not released.
- Poses
  - To test the trained model (`bash ./scripts/pose/test.sh`):
    ```bash
    python test.py --name pose --dataset_mode fewshot_pose --adaptive_spade --warp_ref --spade_combine --remove_face_labels --finetune --seq_path [PATH_TO_SEQ] --ref_img_path [PATH_TO_REF_IMG]
    ```
  
- Faces
  - To test the model (`bash ./scripts/face/test.sh`):
    ```bash
    python test.py --name face --dataset_mode fewshot_face --adaptive_spade --warp_ref --spade_combine --seq_path [PATH_TO_SEQ] --ref_img_path [PATH_TO_REF_IMG]
    ```

- Street
  - To test the model (`bash ./scripts/street/test.sh`):
    ```bash
    python test.py --name street --dataset_mode fewshot_street --adaptive_spade --loadSize 512 --fineSize 512 --seq_path [PATH_TO_SEQ]--ref_img_path [PATH_TO_REF_IMG]
    ```
    

## More Training/Test Details
- Difference of training methodology vs. vid2vid: instead of copying frames from one GPU to another, each GPU now handles separate batches. To fit into memory, the network only generates one frame at a time (n_frames_per_gpu = 1), and keep this frame fixed when generating the next frame in the sequence. We found this is usually sufficient to modify the current frame only, and is more efficient and easier to maintain.
- Training schedule: after switching to using SPADE, the network now consists of two sub-networks: one for single image generation (the SPADE generator) and the flow estimation network. By default, the training will start with training the single frame generator only (i.e. n_frames_total = 1) for `niter_single` epochs. After that, the network will start to train the flow network to generate videos, and temporal losses are introduced. Similar to vid2vid, we double the training sequence length for every `niter_step` epochs after starting training videos.

- Important flags regarding network arch:
  - `adaptive_spade`: adaptively generate network weights for SPADE modules.
  - `no_adaptive_embed`: do not dynamically generate weights for the label embedding network.
  - `n_adaptive_layers`: number of adaptive layers in the generator.
  - `warp_ref`: add an additional flow network to warp the reference image to the current frame and combine with it.
  - `spade_combine`: instead of linearly blending hallucinated and warped frames to generate the final frame, use warped frame as a guidance image in an additional SPADE module during the synthesis process.

- Important flags regarding training:
  - `n_frames_G`: the number of input frames to feed into the generator network; i.e., `n_frames_G - 1` is the number of frames we look into the past.
  - `n_frames_total`: the total number of frames in a sequence we want to train with. We gradually increase this number during training.
  - `niter_single`: the number of epochs we train the single frame generator before starting training videos.
  - `niter_step`: for how many epochs do we double `n_frames_total`. The default is 10.  
  - `batchSize`: the number of training batches. If it is not divisible by number of GPUs, the first GPU (which is usually more memory heavy) will do fewer batches.

- For other flags, please see `options/train_options.py` and `options/base_options.py` for all the training flags; see `options/test_options.py` and `options/base_options.py` for all the test flags.

- Additional flags for pose examples:
  - `remove_face_labels`: remove densepose results for face, so the network can get more robust during inference on different subjects.
  - `basic_point_only`: if specified, only use basic joint keypoints for OpenPose output, without using any hand or face keypoints.
  - `add_face_D`: add an additional discriminator that only works on the face region.  
  - `refine_face`: add an additional network to refine the face region.   

- Additional flags for face examples:
  - `no_upper_face`: by default, we add artificial edges for the upper part of face by symmetry. This flag disables it.

## Citation

If you find this useful for your research, please cite the following paper.

```
@inproceedings{wang2019fewshotvid2vid,
   author    = {Ting-Chun Wang and Ming-Yu Liu and Andrew Tao 
                and Guilin Liu and Jan Kautz and Bryan Catanzaro},
   title     = {Few-shot Video-to-Video Synthesis},
   booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},   
   year      = {2019},
}
```

## Acknowledgments
We thank Karan Sapra for generating the segmentation maps for us.</br>
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and [vid2vid](https://github.com/NVIDIA/vid2vid).
