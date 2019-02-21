import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

from utils.module_3d import Unit3D, STConv3d


__all__ = ['resnet50', 'resnet101', 'resnet152']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, inflat_mode=0):
        super(Bottleneck, self).__init__()
        if inflat_mode == 0:
            self.conv1 = Unit3D(inplanes, planes, kernel_shape=[1, 1, 1])
            self.conv2 = Unit3D(planes, planes, kernel_shape=[1, 3, 3], stride=[1, stride, stride])
            self.conv3 = Unit3D(planes, planes * 4, kernel_shape=[1, 1, 1], activation_fn=None)

        elif inflat_mode == 1:
            self.conv1 = Unit3D(inplanes, planes, kernel_shape=[1, 1, 1])
            self.conv2 = Unit3D(planes, planes, kernel_shape=[3, 3, 3], stride=[1, stride, stride])
            self.conv3 = Unit3D(planes, planes * 4, kernel_shape=[1, 1, 1], activation_fn=None)

        elif inflat_mode == 2:
            self.conv1 = Unit3D(inplanes, planes, kernel_shape=[3, 1, 1], stride=[1, 1, 1])
            self.conv2 = Unit3D(planes, planes, kernel_shape=[1, 3, 3], stride=[1, stride, stride])
            self.conv3 = Unit3D(planes, planes * 4, kernel_shape=[3, 1, 1], stride=[1, 1, 1], activation_fn=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, inflat_mode=[]):
        self.inplanes = 64
        self.inflat_mode = inflat_mode

        super(ResNet, self).__init__()
        self.conv1 = Unit3D(3, 64, kernel_shape=[5, 7, 7], stride=[2, 2, 2])
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=[3, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0])

        self.layer1 = self._make_layer(block, 64, layers[0], inflat_mode=inflat_mode[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, inflat_mode=inflat_mode[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, inflat_mode=inflat_mode[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, inflat_mode=inflat_mode[3])
        self.avgpool = nn.AvgPool3d(kernel_size=[4, 7, 7], stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, inflat_mode=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Unit3D(self.inplanes, planes * block.expansion,
                          kernel_shape=[1, 1, 1], stride=[1, stride, stride])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, inflat_mode))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, inflat_mode=inflat_mode))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.train: # test mode. x.size() = (B, C, T*10, H, W)
            B, C, T, H, W = x.size()
            x = x.transpose(2, 1).view(B*10, -1, C, H, W).transpose(2, 1).contiguous() # (B*10, C, T, H, W)
        x = self.conv1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.mean(x, 2).mean(2).mean(2) # B x num_classes
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def replace_logits(self, num_classes):
        in_channels = self.fc.weight.size()[1]
        self.fc = nn.Linear(in_channels, num_classes)

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    inflat_mode = [0, 0, 2, 2]
    model = ResNet(Bottleneck, [3, 4, 6, 3], inflat_mode=inflat_mode, **kwargs)

    return model



def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    inflat_mode = [0, 0, 2, 2]
    model = ResNet(Bottleneck, [3, 4, 23, 3], inflat_mode=inflat_mode, **kwargs)
    return model



def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    inflat_mode = [0, 0, 2, 2]
    model = ResNet(Bottleneck, [3, 8, 36, 3], inflat_mode=inflat_mode, **kwargs)

    return model


def make_i3dResnet(arch='resnet50', pretrained=False):
    assert arch in ['resnet50', 'resnet101', 'resnet152'], 'illegal architecture'
    model = globals()[arch]()
    if pretrained:
        pretrained_resnet = getattr(torchvision.models, arch)(pretrained=True)
        pretrained_dict = pretrained_resnet.state_dict()
        model_dict = model.state_dict()
        for k,v in model_dict.items():
            if 'conv' in k:
                # print(k)
                weight_trained = pretrained_dict[k]
                size = model_dict[k].size()
                pretrained_dict[k] = pretrained_dict[k].unsqueeze(2).expand(*size).div(size[2])
            elif 'downsample' in k:
                if len(pretrained_dict[k].size()) == 4:
                    weight_trained = pretrained_dict[k]
                    size = model_dict[k].size()
                    pretrained_dict[k] = pretrained_dict[k].unsqueeze(2).expand(*size).div(size[2])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


if __name__=='__main__':
    model = make_i3dResnet(inflat_mode=1)
    input = torch.Tensor(3, 3, 32, 224, 224)
    output = model(input)
    print(output.size())
    # print(model)
    # print(model)

