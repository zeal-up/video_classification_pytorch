import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from pt_dataset.tsndataset import TSNDataSet
from models.tsn_model.models import TSN
from utils.transforms import *
from utils.Trainer import Trainer_cls
from utils.Plot import Visdom_Plot
from opts import parser

cudnn.benchmark = True


def main():
    global args, device
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # visdom
    viz = Visdom_Plot(port=args.visdom_port, env_name=args.visdom_name)
    args_text = str(vars(args))
    viz.append_text(args_text.replace(', \'', '<br>\''), win_name='args_win')

    if args.dataset == 'UCF101':
        num_class = 101
    elif args.dataset == 'HMDB51':
        num_class = 51
    elif args.dataset == 'KINETICS':
        num_class = 400
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality == 'Flow':
        data_length = 10

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
    train_transform = torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       IdentityTransform(),
                   ])

    test_transform = torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       IdentityTransform(),
                   ])

    train_dataset = TSNDataSet(num_segments=args.num_segments, dataset=args.dataset,
                   new_length=data_length, modality=args.modality,
                   transform=train_transform, split=args.split, train=True)

    test_dataset = TSNDataSet(num_segments=args.num_segments, dataset=args.dataset,
                   new_length=data_length, modality=args.modality,
                   transform=test_transform, split=args.split, train=False)
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # define lr_schedular
    lr_schedular = tsn_schedular(
        optimizer,
        lr_steps = args.lr_steps,
        initial_lr = args.lr,
        weight_decay = args.weight_decay
    )
    

    # train
    trainer = Trainer_cls(
        model,
        criterion,
        optimizer,
        train_loader,
        device,
        None,
        viz
    )

    trainer.train(
        args.epochs,
        test_loader=val_loader,
        loader_fn=None,
        lr_scheduler=lr_schedular,
        scheduler_metric=None,
        bn_scheduler=None,
        saved_path=os.path.join('./saved_model', args.visdom_name),
        val_interval=5
    )


    
class tsn_schedular(object):
    def __init__(self, optimizer, lr_steps, initial_lr, weight_decay):
        self.optimizer = optimizer
        self.lr_steps = lr_steps
        self.epoch = 0
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay

    def step(self):
        decay = 0.1 ** (sum(self.epoch >= np.array(self.lr_steps)))
        lr = self.initial_lr * decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']

        self.epoch += 1


if __name__ == '__main__':
    main()
