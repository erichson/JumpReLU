from __future__ import absolute_import

import torch.nn as nn
import math

from activationfun import *



# Resnet for cifar dataset. Adapted from https://github.com/bearpaw/pytorch-classification
def conv3x3(in_planes, out_planes, stride = 1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsample = None, jump=0.0):
        super(BasicBlock, self).__init__()
        
        self.jump = jump
        self.JumpReLU = JumpReLU(jump=self.jump)
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.JumpReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.JumpReLU(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride = 1, downsample = None, jump=0.0):
        super(Bottleneck, self).__init__()
        
        self.jump = jump
        self.JumpReLU = JumpReLU(jump=self.jump)
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = stride,
                               padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        #self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.JumpReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.JumpReLU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu_jump(out)

        return out

ALPHA_ = 1
class JumpResNet(nn.Module):

    def __init__(self, depth, num_classes = 10, jump=0.0):
        super(JumpResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >= 44 else BasicBlock

        self.jump = jump
        self.JumpReLU = JumpReLU(jump=self.jump)

        self.inplanes = 16 * ALPHA_
        self.conv1 = nn.Conv2d(3, 16 * ALPHA_, kernel_size = 3, padding = 1,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(16 * ALPHA_)
        #self.relu = nn.ReLU(inplace = True)
        self.layer1 = self._make_layer(block, 16 * ALPHA_, n, jump = self.jump)
        self.layer2 = self._make_layer(block, 32 * ALPHA_, n, stride = 2, jump = self.jump)
        self.layer3 = self._make_layer(block, 64 * ALPHA_, n, stride = 2, jump = self.jump)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * ALPHA_* block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride = 1, jump = 0.0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, jump = jump))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, jump = jump))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.JumpReLU(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x