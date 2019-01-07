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

args = parser.parse_args()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(10)
torch.cuda.manual_seed(10)
np.random.seed(10)

def main():
    global args, device
    
    # visdom
    viz = Visdom_Plot(port=args.visdom_port, env_name=args.visdom_name)
    args_text = str(vars(args))
    viz.append_text(args_text.replace(', \'', '<br>\''), win_name='args_win')


####### 把另外两个数据集插入config！！！！！！！！！！！！！！！！！！！！！！！！

    if args.model == 'tsn':
        import models.tsn_model.config as config 
    elif args.model == 'i3d':
        import models.I3D.config as config

    model = config.model
    model.to(device)
    train_loader = config.train_loader
    val_loader = config.val_loader

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    
    print('prepare finished, start training')
    
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
        scheduler_metric='best_val_loss',
        bn_scheduler=None,
        saved_path=os.path.join('./saved_model', args.visdom_name),
        val_interval=1
    )



# # define lr_schedular
#         lr_schedular = tsn_schedular(
#             optimizer,
#             lr_steps = args.lr_steps,
#             initial_lr = args.lr,
#             weight_decay = args.weight_decay
#         )
# class tsn_schedular(object):
#     def __init__(self, optimizer, lr_steps, initial_lr, weight_decay):
#         self.optimizer = optimizer
#         self.lr_steps = lr_steps
#         self.epoch = 0
#         self.initial_lr = initial_lr
#         self.weight_decay = weight_decay

#     def step(self):
#         decay = 0.1 ** (sum(self.epoch >= np.array(self.lr_steps)))
#         lr = self.initial_lr * decay
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = lr * param_group['lr_mult']
#             param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']

#         self.epoch += 1


if __name__ == '__main__':
    main()
