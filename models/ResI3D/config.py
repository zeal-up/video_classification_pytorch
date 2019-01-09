import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn

import utils.transforms as ut_transforms 
from pt_dataset.consecutive_dataset import Consecutive
from models.ResI3D.i3dResnet import make_i3dResnet
import main
args = main.args

# 暂时只支持RGB frames 和kinetics\ucf101数据集

train_transforms = T.Compose([
    ut_transforms.GroupRandomScale(size_low=256, size_high=320), # randomly resize smaller edge to [256, 320]
    ut_transforms.GroupRandomCrop(224), # randomlly crop a 224x224 patch
    ut_transforms.GroupToTensor(),
    # ut_transforms.GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ut_transforms.StackTensor()
])

val_transforms = T.Compose([
    ut_transforms.GroupScale(256), # scale to 256 and do fully-convolutional
    ut_transforms.GroupToTensor(),
    # ut_transforms.GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ut_transforms.StackTensor()
])

train_dataset = Consecutive(dataset=args.dataset, train=True, interval=2, transform=train_transforms) #default 64 frames
val_dataset = Consecutive(dataset=args.dataset, train=False, interval=2, transform=val_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, 
    shuffle=True, num_workers=args.workers,
    pin_memory=True)

val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=args.workers,
    pin_memory=True
)

num_class = train_dataset.num_classes

model = make_i3dResnet(arch=args.arch, pretrained=False, inflat_mode=1) # only RGB model avaliable right now

model.replace_logits(num_class)

# model = nn.DataParallel(model)


