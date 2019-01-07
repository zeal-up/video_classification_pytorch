import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn

import utils.transforms as ut_transforms 
from pt_dataset.kinetics_dataset import Kinetics
from models.I3D.I3D_model import InceptionI3d
import main
args = main.args

# 暂时只支持RGB frames 和kinetics数据集

class I3Dscale(object):
    # rescale piexls values in[0, 1] to [-1, 1]
    def __init__(self):
        return

    def __call__(self, data):
        return data*2 - 1.0

if args.dataset == 'UCF101':
    num_class = 101
    pretrained = True
elif args.dataset == 'HMDB51':
    num_class = 51
    pretrained = True
elif args.dataset == 'KINETICS':
    num_class = 400
    pretrained = False
else:
    raise ValueError('Unknown dataset ' + args.dataset)


model = InceptionI3d(400, in_channels=3) # only RGB model avaliable right now
if pretrained:
    model.load_state_dict(torch.load('./models/I3D/rgb_imagenet.pt'))
    model.replace_logits(num_class)
else:
    model.load_state_dict(torch.load('./models/I3D/rgb_imagenet.pt'))
    model.replace_logits(num_class)

model = nn.DataParallel(model)


train_transforms = T.Compose([
    ut_transforms.GroupScale(256), # resize smaller edge to 256
    ut_transforms.GroupRandomCrop(224), # randomlly crop a 224x224 patch
    ut_transforms.GroupRandomHorizontalFlip(),
    ut_transforms.GroupStackToTensor(),
    I3Dscale()
])

val_transforms = T.Compose([
    ut_transforms.GroupCenterCrop(224), # center crop 224x224 patch
    ut_transforms.GroupStackToTensor(),
    I3Dscale()
])

train_dataset = Kinetics(mode='train', transform=train_transforms) #default 64 frames
val_dataset = Kinetics(mode='val', transform=val_transforms)

train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, 
    shuffle=True, num_workers=args.workers,
    pin_memory=True)

val_loader = DataLoader(
    val_dataset, batch_size=args.batch_size,
    shuffle=True, num_workers=args.workers,
    pin_memory=True
)
