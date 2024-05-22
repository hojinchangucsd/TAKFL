import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                        stride=stride, padding=1, bias=False)

def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

def branchBottleNeck(channel_in, channel_out, kernel_size):
    middle_channel = channel_out//4
    return nn.Sequential(
        nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=kernel_size),
        nn.BatchNorm2d(middle_channel),
        nn.ReLU(),
        
        nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class BottleneckBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, planes*self.expansion)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output
    
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

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        output += residual
        output = self.relu(output)
        return output
    
class ResNet(nn.Module):
    """Resnet model

    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, scaling=1.0, num_classes=1000):
        super(ResNet, self).__init__()
        assert int(64 * scaling) > 0
        
        self.inplanes = int(64 * scaling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
        #kernel_size=7, stride=2, padding=3
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        self._make_layer(block, int(64 * scaling), layers[0])
        self._make_layer(block, int(128 * scaling), layers[1], stride=2)
        self._make_layer(block, int(256 * scaling), layers[2], stride=2)
        self._make_layer(block, int(512 * scaling), layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(512 * scaling) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            self.layers.append(block(self.inplanes, planes))
        
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out
    
class ResNet8(nn.Module):
    """Resnet model

    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, scaling=1.0, num_classes=1000):
        super(ResNet8, self).__init__()
        assert int(64 * scaling) > 0
        
        self.inplanes = int(64 * scaling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
        #kernel_size=7, stride=2, padding=3
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        self._make_layer(block, int(64 * scaling), layers[0])
        self._make_layer(block, int(128 * scaling), layers[1], stride=2)
        self._make_layer(block, int(256 * scaling), layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(256 * scaling) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            self.layers.append(block(self.inplanes, planes))
        
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out

class ResNet6(nn.Module):
    """Resnet model

    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, scaling=1.0, num_classes=1000):
        super(ResNet6, self).__init__()
        assert int(64 * scaling) > 0
        
        self.inplanes = int(64 * scaling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
        #kernel_size=7, stride=2, padding=3
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        self._make_layer(block, int(64 * scaling), layers[0])
        self._make_layer(block, int(128 * scaling), layers[1], stride=2)
        #self._make_layer(block, int(256 * scaling), layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(128 * scaling) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            self.layers.append(block(self.inplanes, planes))
        
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out

class ResNet4(nn.Module):
    """Resnet model

    Args:
        block (class): block type, BasicBlock or BottleneckBlock
        layers (int list): layer num in each block
        num_classes (int): class num
    """

    def __init__(self, block, layers, scaling=1.0, num_classes=1000):
        super(ResNet4, self).__init__()
        assert int(64 * scaling) > 0
        
        self.inplanes = int(64 * scaling)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) 
        #kernel_size=7, stride=2, padding=3
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layers = nn.ModuleList()
        self._make_layer(block, int(64 * scaling), layers[0])
        #self._make_layer(block, int(128 * scaling), layers[1], stride=2)
        #self._make_layer(block, int(256 * scaling), layers[2], stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(int(128 * scaling) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d): 
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, layers, stride=1):
        """A block with 'layers' layers

        Args:
            block (class): block type
            planes (int): output channels = planes * expansion
            layers (int): layer num in the block
            stride (int): the first layer stride in the block
        """
        downsample = None
        if stride !=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        self.layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            self.layers.append(block(self.inplanes, planes))
        
        return
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out

if __name__ == "__main__":
    import torch
    net = ResNet8(BasicBlock, [1,1,1], scaling=1.0, num_classes=10)
    net = ResNet6(BasicBlock, [1,1], scaling=1.0, num_classes=10)
    net = ResNet4(BasicBlock, [1], scaling=1.0, num_classes=10)

    print(net)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y.shape)
    
    total = 0
    for name, param in net.named_parameters():
        #print(name, param.size())
        total += np.prod(param.size())
        #print(np.array(param.data.cpu().numpy().reshape([-1])))
        #print(isinstance(param.data.cpu().numpy(), np.array))
    print(f'total params {total}')