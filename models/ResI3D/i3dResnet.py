import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import torch.utils.model_zoo as model_zoo



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, inflat=False):
    """3x3 convolution with padding"""
    kernel_size = [3, 3, 3] if inflat else 3
    padding = [1, 1, 1] if inflat else [0, 1, 1]
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, inflat_mode=0):
        super(Bottleneck, self).__init__()
        if inflat_mode == 0:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=[1, 3, 3], stride=[1, stride, stride],
                                padding=[0, 1, 1], bias=False)
            self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)

        elif inflat_mode == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=[3, 3, 3], stride=[1, stride, stride],
                                   padding=[1, 1, 1], bias=False)
            self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)

        elif inflat_mode == 2:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=[3, 1, 1], stride=[1, 1, 1],
                                    padding=[1, 0, 0], bias=False)
            self.conv2 = nn.Conv3d(planes, planes, kernel_size=[1, 3, 3], stride=[1, stride, stride],
                                    padding=[0, 1, 1], bias=False)
            self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=[3, 1, 1], stride=[1, 1, 1],
                                    padding=[1, 0, 0], bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, inflat_mode=0):
        self.inplanes = 64
        self.inflat_mode = inflat_mode

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=[5, 7, 7], stride=2, padding=[2, 3, 3],
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool3d(kernel_size=[3, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0])

        inflat_count = 0 # inflat every 2 res block

        self.layer1, inflat_count = self._make_layer(block, 64, layers[0], inflat_count=inflat_count)
        self.layer2, inflat_count = self._make_layer(block, 128, layers[1], stride=2, inflat_count=inflat_count)
        self.layer3, inflat_count = self._make_layer(block, 256, layers[2], stride=2, inflat_count=inflat_count)
        self.layer4, inflat_count = self._make_layer(block, 512, layers[3], stride=2, inflat_count=inflat_count)
        self.avgpool = nn.AvgPool3d(kernel_size=[4, 7, 7], stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, inflat_count=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=[1, stride, stride], bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        if inflat_count % 2 == 0 :
            layers.append(block(self.inplanes, planes, stride, downsample, self.inflat_mode))
            inflat_count += 1
        else :
            layers.append(block(self.inplanes, planes, stride, downsample, 0))
            inflat_count += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if inflat_count % 2 == 0 :
                layers.append(block(self.inplanes, planes, inflat_mode=self.inflat_mode))
                inflat_count += 1
            else:
                layers.append(block(self.inplanes, planes, inflat_mode=0))
                inflat_count += 1


        return nn.Sequential(*layers), inflat_count

    def forward(self, x):
        if not self.train: # test mode. x.size() = (B, C, T*10, H, W)
            B, C, T, H, W = x.size()
            x = x.transpose(2, 1).view(B*10, -1, C, H, W).transpose(2, 1).contiguous() # (B*10, C, T, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.maxpool2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x) 

        if not self.train: # x.size() = (B*10, num_classes)
            x = x.view(B, 10, -1)
            x = torch.mean(x, 1)

        return x

    def replace_logits(self, num_classes):
        in_channels = self.fc.weight.size()[1]
        self.fc = nn.Linear(in_channels, num_classes)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def make_i3dResnet(arch='resnet50', pretrained=True, inflat_mode=0):
    assert arch in ['resnet50', 'resnet101', 'resnet152'], 'illegal architecture'
    model = globals()[arch](pretrained=False, inflat_mode=inflat_mode)
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

