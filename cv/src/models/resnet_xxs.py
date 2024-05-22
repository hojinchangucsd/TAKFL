# https://github.com/anilkagak2/DiSK_Distilling_Scaffolded_Knowledge/blob/main/models.py
# SCAFFOLDING A STUDENT TO INSTILL KNOWLEDGE
# 
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_xxs(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_xxs, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_xxs(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_xxs, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                            stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                            planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NewResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, all_planes=[64, 128, 256, 512], adaptive_pool=False):
        super(NewResNet, self).__init__()
        self.in_planes = all_planes[0] #64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, all_planes[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, all_planes[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, all_planes[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( all_planes[3] *block.expansion, num_classes)

        self.adaptive_pool = adaptive_pool
        if self.adaptive_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.conv_channels = [all_planes[2]]
        self.xchannels   = [all_planes[3] * block.expansion] 

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_message(self):
        return 'ResNet Scalability (CIFAR)'#self.message

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.layer4(out)

        if self.adaptive_pool:
            out = self.avg_pool(out)
        else:
            out = F.avg_pool2d(out, 4)
            
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet10_l(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[32, 64, 128, 256], adaptive_pool=adaptive_pool)

def ResNet10_m(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[16, 32, 64, 128], adaptive_pool=adaptive_pool)

def ResNet10_s2(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[16, 16, 32, 64], adaptive_pool=adaptive_pool)

def ResNet10_s(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[8, 16, 32, 64], adaptive_pool=adaptive_pool)

def ResNet10_xs(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[8, 16, 16, 32], adaptive_pool=adaptive_pool)

def ResNet10_xxs(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[8, 8, 16, 16], adaptive_pool=adaptive_pool)

def ResNet10_xxxs(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, all_planes=[4, 8, 8, 16], adaptive_pool=adaptive_pool)

def ResNet10(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [1, 1, 1, 1], num_classes=num_classes, adaptive_pool=adaptive_pool)

def ResNet18(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [2, 2, 2, 2], num_classes=num_classes, adaptive_pool=adaptive_pool)

def ResNet34(num_classes=100, adaptive_pool=False):
    return NewResNet(BasicBlock_xxs, [3, 4, 6, 3], num_classes=num_classes, adaptive_pool=adaptive_pool)

def ResNet50(num_classes=100, adaptive_pool=False):
    return NewResNet(Bottleneck_xxs, [3, 4, 6, 3], num_classes=num_classes, adaptive_pool=adaptive_pool)