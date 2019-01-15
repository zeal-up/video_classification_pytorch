# video_classification_pytorch
This repository includes some classical network architecture of video classification(action recognition). Because of the scale of Kinetics,  most of the architectures in this repo have not be tested on kenetics. But the training loss curve seems normal in the training procedure.
This project can be regard as a gather of some implementations by PyTorch of the corresponding paper.


## 1. Requirements

PyTorch1.0
visdom (the training procedure can be track in explorer)
PIL

## 2. Install 

To make sure all the sub_modules is clone correctly, use
```
git clone https://github.com/zeal-github/video_classification_pytorch.git
```

## 3. Prepare datasets

All the data list is pre-computed in the `.json` file, and the dataset loader will first load the `.json` file
as the avaliable data samples. Up to now, `UCF101`, `Kinetics_400`, `Kinetics_200` are all avaliable in the dataset loader.
Notice: so far, only RGB frames is consider in this repo.

### 3.1 UCF101

You can download the preprocessed data directly from [https://github.com/feichtenhofer/twostreamfusion]

```
cd ./videos_dataset/UCF101
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003

cat ucf101_jpegs_256.zip* > jpegs_256.zip
unzip jpegs_256.zip
```

### 3.2 Kinetics

Since the Kinetics is too large and we can only download the data use the official crawler.
You can refer to this repo, download the `.avi` data using the official crawler and
convert the videos data to `.jpg` frames.
You can use code in `./pt_dataset` to create a `json` data which contains the datalist of the 
training and validation data. Each element in `datalist.json` files contains 4 items:
['path':frame directory, 'class_name':classname, 'label':label, 'num_frames':number of frames of this video].

## 4. Avaliable network:

All the networks and the pretrained models are contained in the `./models`. 

### 4.1 I3D 

Original paper: "[Quo Vadis,Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750)"
This code is based on : [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d)

The pretrained model is provided by [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d). The pretrained model
is pretrained on `ImageNet` and `Kinetics` as reported in the paper.

### 4.2 S3D

Original paper: "[Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification](https://link.springer.com/chapter/10.1007/978-3-030-01267-0_19)
This code is based on : [qijiezhao/s3d.pytorch](https://github.com/qijiezhao/s3d.pytorch)

The pretrained model is the model in `4.1 I3D`. Only the 2D filters will transfer parameters from the pretrained `I3D` model.

### 4.3 3D ResNet

Original paper: "[Non-local Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)"

To stay the same as the original paper, I inflate the `Residual block` for every two `Residual block` to saving the computation cost.
3 inflate mode can be choosed(must mannualy specify in config.py):
```
0 : is the baseline model in the paper.
1 : inflate the 1x1 convolution in Residual block to 3x1x1 convolution
2 : inflate the 1x1 convolution in Residual block to 3x3x3 convolution
```

### 4.4 TSN

Original paper: "[Temporal Segment Networks for Action Recognition in Videos](https://ieeexplore.ieee.org/abstract/document/8454294)"
This code is mostly copy from [yjxiong/tsn-pytorch](https://github.com/yjxiong/tsn-pytorch)


## 5. Training options

All the avaliable options in contains in `./opts.py` file. All the options can be specify as argument when start training use `python main.py`.
By default, all gpus will be used for training. To specify avaliable GPU, use:
```
CUDA_AVALIBLE_DEIVDES=0,1...
```

# Ask for contributors

If you are a researcher of video learning and is interesting in share some code. It's welcome to pull request. If you find come unreasonable 
arangement in this code, or some new architecture you want, just raise a issue.