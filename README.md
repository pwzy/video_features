
<div align="center">

# Video Features

<img  src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/video_features/vid_feats.gif" width="300" />

`video_features` allows you to extract features from video clips.
It supports a variety of extractors and modalities,
i.e. visual appearance, optical flow, and audio.
See more details in [Documentation](https://v-iashin.github.io/video_features/).
</div>

## Supported Models

Action Recognition

- [S3D (Kinetics 400)](https://v-iashin.github.io/video_features/models/s3d)
- [R(2+1)d RGB (IG-65M, Kinetics 400)](https://v-iashin.github.io/video_features/models/r21d)
- [I3D-Net RGB + Flow (Kinetics 400)](https://v-iashin.github.io/video_features/models/i3d)

Sound Recognition

- [VGGish (AudioSet)](https://v-iashin.github.io/video_features/models/vggish)

Optical Flow

- [RAFT (FlyingChairs, FlyingThings3D, Sintel, KITTI)](https://v-iashin.github.io/video_features/models/raft)
- [PWC-Net (Sintel)](https://v-iashin.github.io/video_features/models/pwc/)

Frame-wise Features

- [CLIP](https://v-iashin.github.io/video_features/models/clip)
- [ResNet-18,34,50,101,152 (ImageNet)](https://v-iashin.github.io/video_features/models/resnet)


## Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1csJgkVQ3E2qOyVlcOM-ACHGgPBBKwE2Y?usp=sharing)

or run with `conda` locally:
```bash
# clone the repo and change the working directory
git clone https://github.com/v-iashin/video_features.git
cd video_features

# install environment
conda env create -f conda_env_torch_zoo.yml

# load the environment
conda activate torch_zoo

# extract r(2+1)d features for the sample videos
python main.py \
    feature_type=r21d \
    device="cuda:0" \
    video_paths="[./sample/v_ZNVhz7ctTq0.mp4, ./sample/v_GGSY1Qvo990.mp4]"

# if you have many GPUs, just run this command from another terminal with another device
# device can also be "cpu"
```

If you are more comfortable with Docker, there is a
Docker image with a pre-installed environment that supports all models.
Check out the
[Docker support](https://v-iashin.github.io/video_features/meta/docker).
documentation page.


## Multi-GPU and Multi-Node Setups

With `video_features`, it is easy to parallelize feature extraction among many GPUs.
It is enough to start the script in another terminal with another GPU (or even the same one)
pointing to the same output folder and input video paths.
The script will check if the features already exist and skip them.
It will also try to load the feature file to check if it is corrupted (i.e. not openable).
This approach allows you to continue feature extraction if the previous script failed for some reason.

If you have an access to a GPU cluster with shared disk space you may scale
extraction with as many GPUs as you can by creating several single-GPU jobs with the same command.

Since each time the script is run the list of input files is shuffled, you don't need to worry that
workers will be processing the same video.
On a rare occasion when the collision happens, the script will rewrite previously extracted features.

## Used in

* [SpecVQGAN](https://arxiv.org/abs/2110.08791) branch `specvqgan`
* [BMT](https://arxiv.org/abs/2005.08271) branch `bmt`
* [MDVC](https://arxiv.org/abs/2003.07758) branch `mdvc`

Please, let me know if you found this repo useful for your projects or papers.

## Acknowledgements

- [@Kamino666](https://github.com/Kamino666): added CLIP model as well as Windows and CPU support (and
many other small things).
- [@borijang](https://github.com/borijang): for solving bugs with file names, I3D checkpoint loading enhancement and code style improvements.
- [@ohjho](https://github.com/ohjho): added support of 37-layer R(2+1)d favors.


## Extract UCF-Crime Features
获得视频地址：
```bash
rm sample/sample_video_paths.txt
find /home/jing/project/dataset/UCF-Crime-unzip  -name "*mp4" > sample/sample_video_paths.txt 
```
进行特征提取：
```python
python main.py \
    feature_type=i3d \
    device="cuda:0" \
    stack_size=16 \
    step_size=16 \
    file_with_video_paths=./sample/sample_video_paths.txt \
    on_extraction=save_numpy \
    output_path="./output"
```
注意：
1. 参数不可指定extraction_fps，因为视频进行重新帧率编码后可能会增加多余的帧。例如14帧的视频进行重新编码后帧数变为21。
2. i3d提取过程中至少需要17帧才能生成特征（因为需要多一帧计算光流）。
3. 提取过程中可进行文件拆分，拆分后文件为split_file_aa, split_file_ab,...
```bash
split -l 1000 example.txt split_file_
```
4. ShanghaiTech dataset数据集的帧率为24，但在进行特征提取时不要修改帧率参数，可能导致视频重新编码后视频总帧数增加。
