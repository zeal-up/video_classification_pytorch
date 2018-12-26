from visdom import Visdom
import torch
import numpy as np 
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Visdom_Plot(object):
    def __init__(self, port=8097, env_name='main'):
        #当‘env_name'不是main时，创建一个新环境
        self.viz = Visdom(port=port, env=env_name)
        self.loss_win = {}
        self.acc_win = {}
        self.text_win = {}
        self.plt_img_win = {}
        self.images_win = {}
        self.gray_win = {}

    def _new_win(self, type='loss_win', win_name='default_loss_win', id='train_loss', H_img=100):
        '''
        type: loss_win, acc_win, text_win, plt_img_win
        name: default is the default win in class. you can specify a window's name
        id: the line's name
        '''

        assert type in ['loss_win', 'acc_win', 'text_win', 'plt_img_win', 'gray_win'], "win type must a string inside  ['loss_win', 'acc_win', 'text_win', 'plt_img_win'] "
        if type == 'loss_win':
            self.loss_win[win_name] = self.viz.line(
                X = np.array([0]),
                Y = np.array([0]),
                name = id,
                opts=dict(
                    xlabel='Epoch.batch',
                    ylabel='Loss',
                    title=win_name,
                    marginleft=60,
                    marginbottom=60,
                    margintop=80,
                    width=800,
                    height=600,
                ))
        elif type == 'acc_win':
            self.acc_win[win_name] = self.viz.line(
                X = np.array([0]),
                Y = np.array([0]),
                name = id,
                opts = dict(
                    xlabel='Epoch.batch',
                    ylabel='Top1 accuracy',
                    title=win_name,
                    showlegend=True,
                    markercolor=np.array([[255, 0, 0]]),
                    marginleft=60,
                    marginbottom=60,
                    margintop=60,
                    width=800,
                    height=600,
                )
            )
        elif type == 'plt_img_win' or type == 'gray_win':
            getattr(self, type)[win_name] = self.viz.images(
                np.random.randn(1, 3, 100, 100),
                opts = dict(
                    height = H_img*5,
                    width = H_img*5,
                )
            )
        elif type == 'text_win':
            self.text_win[win_name] = self.viz.text('Text Window')

    def append_loss(self, loss, epoch_batches, win_name='default_loss_win', id='train_loss'):
        if win_name not in self.loss_win:
            self._new_win(type='loss_win', win_name=win_name, id=id)
        self.viz.line(
            X = np.array([epoch_batches]),
            Y = np.array([loss]),
            win = self.loss_win[win_name],
            name = id,
            opts=dict(showlegend=True),
            update='append'
        )

    def append_acc(self, train_acc, epoch_batches, win_name='default_acc_win', id='train_acc'):
        if win_name not in self.acc_win:
            self._new_win(type='acc_win', win_name=win_name, id=id)

        self.viz.line(
            X = np.array([epoch_batches]),
            Y = np.array([train_acc]),
            win = self.acc_win[win_name],
            name = id,
            opts=dict(showlegend=True),
            update='append'
        )

    def lr_scatter(self, epoch, lr, win_name='default_acc_win'):
        self.viz.scatter(
            X = np.array([[epoch, 20]]),
            name = 'lr=' + str(lr),
            win = self.acc_win[win_name],
            opts=dict(showlegend=True),
            update='append',
        )

    def img_plot(self, images, lm=None, mode='update', caption=''):
        '''
        Input:
        images : tensors, N x 3 x H x W, so transfer to N x H x W x 3 is needed
        lm : N x K x 2, is not None, then landmarks will be scattered.
        '''
        win_exist = len(self.plt_img_win)
        N, C, H, W = images.size()
        if N > win_exist:
            for i in range(win_exist, N, 1):
                self._new_win(type='plt_img_win', win_name='image'+str(i), H_img=H)
        if lm is not None:
            N, K, m = lm.size()
            assert N == images.size()[0] and m == 2, "landmarks have illegal size"
            lm = lm.cpu()
        images = images.cpu()

        plt.figure(figsize=(H*0.06,W*0.06))
        for n,image in enumerate(images[:]):
            # print(image.size())
            image = image.transpose(0, 1).transpose(1,2)
            # print(image.size())
            plt.imshow(image.detach().numpy()) # convert to H x W x 3. plt的输入是HxWx3，而viz.images())的输入是3xHxW
            if lm is not None:
                color = np.linspace(0, 1, num=K)
                plt.scatter(x=lm[n,:,0].detach().numpy(), y=lm[n,:,1].detach().numpy(), c=color, marker='x', s=200)
                self.viz.matplot(plt, win=self.plt_img_win['image'+str(n)], opts=dict(caption='image'+str(n)))
                plt.clf()
    def images(self, images, win_name='default_images_win'):
        '''
        Input:
        images:N x 3 x H x W, tensors
        '''
        images = images.cpu()
        if win_name not in self.images_win:
            self.images_win[win_name] = self.viz.images(images.detach().numpy())
        else:
            self.viz.images(images.detach().numpy(), win=self.images_win[win_name])

    def gray_images(self, images, win_name='default_gray_win'):
        '''
        Input:
        images : K x H x W, tensors
        '''
        images = images.cpu()
        win_exist = len(self.gray_win)
        K, H, W = images.size()
        if K > win_exist:
            for i in range(win_exist, K, 1):
                self._new_win(type='gray_win', win_name='gray'+str(i), H_img=H//2)

        plt.figure(figsize=(H/2*0.06,W/2*0.06))
        for n,image in enumerate(images):
            plt.imshow(image.detach().numpy()) # convert to H x W x 3. plt的输入是HxWx3，而viz.images())的输入是3xHxW
            self.viz.matplot(plt, win=self.gray_win['gray'+str(n)])
            plt.clf()



    def append_text(self, text, win_name='default_text_win', append=True):
        if win_name not in self.text_win:
            self._new_win(type='text_win', win_name=win_name)
        self.viz.text(text, win=self.text_win[win_name], append=append)
        






if __name__ == '__main__':
    viz = Visdom_Plot('test', env_name='new_1')
    loss = np.arange(101, 1, -1)
    train_acc = np.array(np.linspace(10, 100, 100)).reshape(100,)
    valid_acc = train_acc - 9
    N = 2
    images = torch.randn(N, 3, 96, 96)
    lm = np.arange(10) + 10
    lm = np.repeat(lm, 2).reshape(10,2)
    lm = np.array(np.stack([lm]*N))
    print('lm size', lm.shape)
    lm = torch.tensor(lm)
    viz.img_plot(images, lm=lm)

    for batches in range(10):
        viz.append_loss(loss[batches], batches)
        viz.append_acc(train_acc[batches], batches, id='train_acc')
        viz.append_acc(valid_acc[batches], batches, id='valid_acc')
    
    text = '测试用文字'
    viz.append_text(text)