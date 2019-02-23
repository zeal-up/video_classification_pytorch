import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os
import sys



class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        _, _, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Sequential):

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=nn.ReLU(inplace=True),
                 bn=True,
                 bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._bn = bn
        self._bias = bias
        self.name = name
        self.padding = padding
        
        conv_unit = nn.Conv3d(in_channels=in_channels,
                            out_channels=self._out_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                            bias=self._bias)

        nn.init.xavier_normal_(conv_unit.weight)

        self.add_module(name+'_conv', conv_unit)

        if self._bn:
            bn = nn.BatchNorm3d(self._out_channels, eps=0.001, momentum=0.01)
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
            self.add_module(name+'_bn', bn)
        if activation_fn is not None:
            self.add_module(name+'_act_fn', activation_fn)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_size[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_size[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        _, _, t, h, w = x.size()
        #print t,h,w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = super().forward(x)

        return x


class STConv3d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, name='stconv3d'):
        '''
        only cuboid kernel can use this module to seperate spatio-temporal convolution
        '''
        super(STConv3d, self).__init__()

        conv1 = Unit3D(in_channels, out_channels, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=0)
        conv2 = Unit3D(out_channels, out_channels, kernel_size=(kernel_size, 1, 1), stride=(stride, 1, 1), padding=0)
        
        self.add_module(name+'_s_conv', conv1)
        self.add_module(name+'_t_conv', conv2)

