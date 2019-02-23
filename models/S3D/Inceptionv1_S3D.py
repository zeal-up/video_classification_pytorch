import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
import numpy as np
sys.path.append(os.path.abspath('.'))
from utils.utils_3d import Unit3D, MaxPool3dSamePadding, STConv3d

class InceptionModuleI3D(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModuleI3D, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_size=[1, 1, 1],
                         name=name+'/Branch_0/Conv3d_0a_1x1x1')
        self.b1a = Unit3D(in_channels=in_channels, out_channels=out_channels[1], kernel_size=[1, 1, 1],
                          name=name+'/Branch_1/Conv3d_0a_1x1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3x3')
        self.b2a = Unit3D(in_channels=in_channels, out_channels=out_channels[3], kernel_size=[1, 1, 1],
                          name=name+'/Branch_2/Conv3d_0a_1x1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, out_channels=out_channels[5], kernel_size=[1, 1, 1],
                          name=name+'/Branch_3/Conv3d_0b_1x1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)

class InceptionModuleS3D(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModuleS3D, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, out_channels=out_channels[0], kernel_size=[1, 1, 1],
                         name=name+'/Branch_0/Conv3d_0a_1x1x1')

        self.b1a = Unit3D(in_channels=in_channels, out_channels=out_channels[1], kernel_size=[1, 1, 1],
                          name=name+'/Branch_1/Conv3d_0a_1x1x1')
        self.b1b = STConv3d(in_channels=out_channels[1], out_channels=out_channels[2], kernel_size=3,
                          name=name+'/Branch_1/Conv3d_0b_3x3x3')

        self.b2a = Unit3D(in_channels=in_channels, out_channels=out_channels[3], kernel_size=[1, 1, 1],
                          name=name+'/Branch_2/Conv3d_0a_1x1x1')
        self.b2b = STConv3d(in_channels=out_channels[3], out_channels=out_channels[4], kernel_size=3,
                          name=name+'/Branch_2/Conv3d_0b_3x3x3')

        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, out_channels=out_channels[5], kernel_size=[1, 1, 1],
                          name=name+'/Branch_3/Conv3d_0b_1x1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


class InceptionS3d(nn.Module):
    """
    Inception-v1 S3D architecture.
    """
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, in_channels=3, name='s3d'):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          name: A string (optional). The name of this module.
        """

        super(InceptionS3d, self).__init__() 
        self._num_classes = num_classes 
        self.logits = None

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = STConv3d(in_channels=in_channels, out_channels=64, kernel_size=7,
                                              stride=2, name=name+end_point)
        
        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))
        
        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, out_channels=64, kernel_size=[1, 1, 1], padding=0,
                                       name=name+end_point)
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = STConv3d(in_channels=64, out_channels=192, kernel_size=3,
                                       name=name+end_point)

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2))
        
        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModuleS3D(192, [64,96,128,16,32,32], name+end_point)

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModuleS3D(256, [128,128,192,32,96,64], name+end_point)

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2))

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModuleS3D(128+192+96+64, [192,96,208,16,48,64], name+end_point)

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModuleS3D(192+208+48+64, [160,112,224,24,64,64], name+end_point)

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModuleS3D(160+224+64+64, [128,128,256,24,64,64], name+end_point)

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModuleS3D(128+256+64+64, [112,144,288,32,64,64], name+end_point)

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModuleS3D(112+288+64+64, [256,160,320,32,128,128], name+end_point)

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2))

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModuleS3D(256+320+128+128, [256,160,320,32,128,128], name+end_point)

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModuleS3D(256+320+128+128, [384,192,384,48,128,128], name+end_point)

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.logits = Unit3D(in_channels=384+384+128+128, out_channels=self._num_classes,
                             kernel_size=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             bn=False,
                             bias=True,
                             name='logits')

        self.build()


    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, out_channels=self._num_classes,
                             kernel_size=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             bn=False,
                             bias=True,
                             name='logits')
        
    
    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        
    def forward(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x) # use _modules to work with dataparallel

        x = self.avg_pool(x)
        x = self.logits(self.dropout(x))
        x = x.mean(3).mean(3)
        # logits is batch X time X classes, which is what we want to work with
        x = torch.mean(x, 2)
        return x

def make_inception_s3d(num_classes=400):
    # pretrained model is I3D model
    model = InceptionS3d()
        
    if num_classes != 400:
        model.replace_logits(num_classes)

    return model


if __name__ == '__main__':
    model = make_inception_s3d(num_classes=101)
    from utils.torchsummary import model_parameter_number, summary
    model_parameter_number(model)
    print(model)
    summary(model, (3, 32, 224, 224))
    data = torch.Tensor(2, 3, 32, 224, 224)
    output = model(data)

    print(output.size())
