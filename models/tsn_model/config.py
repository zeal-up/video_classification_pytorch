import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os 

import utils.transforms as ut_transforms 
from pt_dataset.tsndataset import TSNDataSet
from models.tsn_model.models import TSN
import main

args = main.args
 
if args.modality == 'RGB':
    data_length = 1
elif args.modality == 'Flow':
    data_length = 10

if args.dataset == 'UCF101':
        num_class = 101
elif args.dataset == 'HMDB51':
    num_class = 51
elif args.dataset == 'KINETICS':
    num_class = 400
else:
    raise ValueError('Unknown dataset ' + args.dataset)

# model definition & configeration parameters
model = TSN(num_class, args.num_segments, args.modality,
            base_model=args.arch, consensus_type=args.consensus_type, 
            dropout=args.dropout, partial_bn=not args.no_partialbn)

crop_size = model.crop_size
scale_size = model.scale_size
input_mean = model.input_mean
input_std = model.input_std
policies = model.get_optim_policies()
train_augmentation = model.get_augmentation()

model = torch.nn.DataParallel(model)
    
if args.no_partialbn:
    model.module.partialBN(False)
else:
    model.module.partialBN(True)

# load model state_dict
if args.resume:
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['acc']
        model.load_state_dict(checkpoint['state_dict'])
        print(("=> loaded checkpoint '{}' (epoch {})"
                .format(args.evaluate, checkpoint['epoch'])))
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume))) 

# dataset
train_transform = T.Compose([
                    train_augmentation,
                    ut_transforms.Stack(roll=args.arch == 'BNInception'),
                    ut_transforms.ToTorchFormatTensor(div=args.arch != 'BNInception'),
                    ut_transforms.IdentityTransform(),
                ])

test_transform = T.Compose([
                    ut_transforms.GroupScale(int(scale_size)),
                    ut_transforms.GroupCenterCrop(crop_size),
                    ut_transforms.Stack(roll=args.arch == 'BNInception'),
                    ut_transforms.ToTorchFormatTensor(div=args.arch != 'BNInception'),
                    ut_transforms.IdentityTransform(),
                ])

train_dataset = TSNDataSet(num_segments=args.num_segments, dataset=args.dataset,
                new_length=data_length, modality=args.modality,
                transform=train_transform, split=args.split, train=True)

test_dataset = TSNDataSet(num_segments=args.num_segments, dataset=args.dataset,
                new_length=data_length, modality=args.modality,
                transform=test_transform, split=args.split, train=False)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

