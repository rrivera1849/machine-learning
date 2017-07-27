
import math
import torch
import torch.nn as nn

class DenseNetLayer(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DenseNetLayer, self).__init__()

    self.bn = nn.BatchNorm2d(in_channels)
    self.relu = nn.ReLU(inplace=True)
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                          stride=1, padding=1, bias=True)

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

class DenseNetBC(nn.Module):
  def __init__(self, num_classes=10, l=100, k=12, theta=0.5):
    super(DenseNetBC, self).__init__()

    num_block_layers = (l - 1 - 3) / 4
    num_block_layers /= 2
    block = DenseNetBottleneckLayer

    self.init_conv = nn.Conv2d(3, 2*k, kernel_size=3, stride=1, padding=1, bias=True)

    self.block1 = DenseNetBlock(block, num_block_layers, 2*k, k)
    in_channels = 2*k + num_block_layers * k
    out_channels = int(math.floor(in_channels * theta))
    self.transition1 = DenseNetTransitionLayer(in_channels, out_channels)
    in_channels = out_channels

    self.block2 = DenseNetBlock(block, num_block_layers, in_channels, k)
    in_channels = in_channels + num_block_layers * k
    out_channels = int(math.floor(in_channels * theta))
    self.transition2 = DenseNetTransitionLayer(in_channels, out_channels)
    in_channels = out_channels

    self.block3 = DenseNetBlock(block, num_block_layers, in_channels, k)
    in_channels = in_channels + num_block_layers * k
    out_channels = int(math.floor(in_channels * theta))
    self.transition3 = DenseNetTransitionLayer(in_channels, out_channels)
    in_channels = out_channels

    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(in_channels, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def forward(self, x):
    out = self.init_conv(x)
    out = self.block1(x)
    out = self.transition1(x)
    out = self.block2(x)
    out = self.transition2(x)
    out = self.block3(x)
    out = self.transition3(x)
    out = self.avgpool(out)
    out = self.classifier(out.view(out.size()[0], -1))

    return out
