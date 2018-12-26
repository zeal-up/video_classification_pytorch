import torch
import os
import time
import traceback
import numpy as np

class Trainer_cls(object):
    '''
    一个通用的用来训练分类网络的类
    lr_scheduler : will update per epoch
    '''
    def __init__(self, model, loss_function, optimizer,\
                train_loader, device=torch.device('cpu'), 
                checkpoint_name=None, viz=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.loss_function = loss_function
        self.viz = viz


        self.best_val_acc = 0



    def train(self, nepochs, test_loader=None, loader_fn=None, lr_scheduler=None, \
                scheduler_metric='best_train_loss', bn_scheduler=None, 
                saved_path='checkepoints/', val_interval=5):
        '''
        loader_fn : 可以用来指定如何使用train_loader 和 test_loader, 默认loader[0]=data, loader[1]=target
                batch_data = next(iter(train_loader))
                lambda batch_data : batch_data[0], batch_data[1]
        '''

        self.log_interval = max(int(len(self.train_loader)/100) , 10)
        # evaluation 100 times per epoch

        try:
            os.makedirs(saved_path)
        except OSError:
            pass

        lr = 0
        train_prec1 = AverageMeter()
        train_prec5 = AverageMeter()
        train_loss = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        try:
            for epoch in range(nepochs):
                self.model.train()
                for batch, batch_data in enumerate(self.train_loader):
                    if loader_fn is not None:
                        data, target = loader_fn(batch_data)
                    else:
                        data, target = batch_data[0], batch_data[1]
                    data, target = data.to(self.device), target.to(self.device)

                    # measure data loading time
                    data_time.update(time.time() - end)

                    #compute output and do SGD step
                    output = self.model(data)
                    loss = self.loss_function(output, target)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # measure accuracy and record loss
                    if batch % self.log_interval == 0:
                        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
                        train_prec1.update(prec1[0], data.size()[0])
                        train_prec5.update(prec5[0], data.size()[0])
                        train_loss.update(loss.item(), data.size()[0])

                        print('Epoch {}: {}/{}  |train_loss:{:.4f}  |train_acc:{:.4F}%'.\
                            format(epoch, batch, len(self.train_loader), loss.item(), train_acc.val))
                        x_axis = round(epoch+batch/len(self.train_loader), 2)
                        self.viz.append_loss(loss.item(), x_axis, win_name='loss_win', id='train_loss')
                        self.viz.append_acc(train_prec1.val, x_axis, win_name='acc_win', id='train_prec1')
                        self.viz.append_acc(train_prec5.val, x_axis, win_name='acc_win', id='train_prec5')

                # log for one epoch
                self.viz.append_text(
                    'Epoch:{}\n'
                    'Datatime:{:.4f}\n'
                    'Batchtime:{:.4f}\n'
                    'TrainLoss:{:.4f}\n'
                    'TrainPrec1:{:4f}\n'
                    'TrainPrec5:{:4f}\n'.format(
                        epoch, data_time.avg, batch_time.avg, 
                        train_loss.avg, train_prec1.avg, train_prec5.avg
                    ),
                    append=False
                )
                train_loss.reset()
                train_prec1.reset()
                train_prec5.reset()
                data_time.reset()
                batch_time.reset()

                # evaluate on validation set
                if (epoch + 1) % val_interval == 0 or epoch == nepochs - 1:
                    val_prec1 = self.validate(test_loader, epoch=epoch+1, loader_fn=loader_fn)                

                    # remember best acc and save checkpoint
                    is_best = val_prec1 > self.best_val_acc
                    self.best_val_acc = max(val_prec1, self.best_val_acc)
                    self._checkpoint(epoch+1, val_prec1, is_best, path=saved_path)

                # adjust learning rate
                if lr_scheduler is not None:
                    if scheduler_metric is None:
                        lr_scheduler.step()
                    else:
                        assert scheduler_metric in ['best_train_loss', 'best_train_acc', 'best_val_loss', 'best_val_acc'],\
                            'illegal schedular metric' 
                        lr_scheduler.step(getattr(self, scheduler_metric))
                    if lr != self.optimizer.param_groups[0]['lr']:
                        lr = self.optimizer.param_groups[0]['lr']
                        self.viz.append_text('\n lr change to {} in epoch {}'.format(lr, epoch))

                # adjust batchnormalization momentum
                if bn_scheduler is not None:
                    bn_scheduler.step()

        except KeyboardInterrupt:
            print('End train early in epoch:', epoch)
            self.viz.append_text('End train early in epoch: ' + str(epoch), win_name='error')
            torch.save(self.model.state_dict(), os.path.join(saved_path, 'end_early_epoch_'+str(epoch)+'.pth'))
        except Exception as E :
            print('Something error: \n', E)
            self.viz.append_text('Something error: ' + str(E) + '\n', win_name='error')
            traceback.print_exc()
        else:
            print('Train finished   |best_val_prec1:{:.4f}'.format(self.best_val_acc))
            self.viz.append_text(\
                'Train finished     |best_val_prec1:{:.4f}'.format(self.best_val_acc),
                win_name='save_information')

    def validate(self, test_loader, epoch, loader_fn=None):
        val_prec1 = AverageMeter()
        val_prec5 = AverageMeter()
        val_loss = AverageMeter()

        with torch.no_grad():
            for batch_data in test_loader:
                self.model.eval()
                if loader_fn is not None:
                    data, target = loader_fn(batch_data)
                else:
                    data, target = batch_data[0], batch_data[1]
                data, target = data.to(self.device), target.to(self.device)

                # compute output
                output = self.model(data)
                loss = self.loss_function(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1,5))

                val_loss.update(loss.data[0], data.size()[0])
                val_prec1.update(prec1[0], data.size()[0])
                val_prec5.update(prec5[0], data.size()[0])

            self.viz.append_loss(val_loss.avg, epoch, win_name='loss_win', id='val_loss')
            self.viz.append_acc(val_prec1.avg, epoch, win_name='acc_win', id='val_prec1')
            self.viz.append_acc(val_prec5.avg, epoch, win_name='acc_win', id='val_prec5')
            print('Epoch {} ||val_loss:{:.4f} |val_prec1:{:.4f} |val_prec5:{:.4f}'.\
                    format(epoch, val_loss.avg, val_prec1.avg, val_prec5.avg))

        return val_prec1.avg

    def _checkpoint(self, epoch, acc, is_best, path='./checkpoint'):
        save_state = {
            'epoch' : epoch,
            'state_dict' : self.model.state_dict(),
            'acc' : acc
        }
        file_name = os.path.join(path, 'epoch{:03d}.pth.tar'.format(epoch))
        torch.save(save_state, file_name)

        if is_best:
            file_name = os.path.join(path, 'best_model.pth.tar')
            torch.save(save_state, file_name)
            self.viz.append_text('save best_model:\n'
                                'Epoch:{:.03d}\n'
                                'acc:{:.4f}\n'.format(epoch, acc), win_name='save_information')




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n 
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res