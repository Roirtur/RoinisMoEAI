import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add(Residual) -> ReLU
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DenseResNet(nn.Module):
    """
    A ResNet-based Dense Baseline.
    Based on ResNet-18 architecture but with adjustable width.
    
    To achieve Iso-Param or Iso-FLOP comparisons with the MoE:
    - Adjust `width_multiplier` (k) to scale the number of filters.
    - k < 1.0 -> Lower capacity (Iso-FLOP baseline potentially).
    - k > 1.0 -> Higher capacity (Iso-Param baseline potentially).
    """
    def __init__(self, input_shape=(3, 32, 32), num_classes=100, width_multiplier=1.0):
        super(DenseResNet, self).__init__()
        self.in_planes = int(64 * width_multiplier)
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_shape[0], self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        # ResNet Layers (Standard ResNet18 layout: 2 blocks per layer)
        # We scale the planes by width_multiplier
        self.layer1 = self._make_layer(BasicBlock, int(64 * width_multiplier), 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, int(128 * width_multiplier), 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, int(256 * width_multiplier), 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, int(512 * width_multiplier), 2, stride=2)
        
        self.linear = nn.Linear(int(512 * width_multiplier) * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def count_parameters(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_baseline(input_shape, num_classes, width_multiplier=1.0):
    return DenseResNet(input_shape=input_shape, num_classes=num_classes, width_multiplier=width_multiplier)
