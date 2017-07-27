
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNetLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DenseNetLayer, self).__init__()

    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                          stride=1, padding=1, bias=True)
    pass

  def forward(self, x):
    out = self.conv(self.relu(self.bn(x)))
    out = torch.cat([x, out], 1)

    return out

class DenseNetBottleneckLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DenseNetBottleneckLayer, self).__init__()

    # out_planes = k
    bottle_channels = 4 * out_planes 
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_channels, bottle_channels, kernel_size=1, 
                           stride=1, padding=0, bias=True)

    self.bn2 = nn.BatchNorm2d(bottle_channels)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(bottle_channels, out_channels, kernel_size=3, 
                           stride=1, padding=1, bias=True)

  def forward(self, x):
    out = self.conv1(self.relu1(self.bn1(x)))
    out = self.conv2(self.relu2(self.bn2(x)))
    out = torch.cat([x, out], 1)

    return out

class DenseNetTransitionLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DenseNetTransitionLayer, self).__init__()

    self.bn = nn.BatchNorm2d(in_channels)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=1, padding=0, bias=True)
    self.pool = nn.AvgPool2d(2)

  def forward(self, x):
    out = self.pool(self.conv(self.bn(x)))

    return out

class DenseNetBlock(nn.Module):
  def __init__(self, block, num_layers, in_channels, k):
    layers = []
    for l in num_layers:
      layers.append(block(in_channels + l * k, k))

    self.dense_block = nn.Sequential(*layers)

  def forward(self, x):
    return self.dense_block(x)
