import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

import torchvision
import torchvision.transforms as transforms

torch.manual_seed(0)
np.random.seed(0)

# Complex multiplication


def compl_mul2d(a, b):
    op = partial(torch.einsum, "bctq,dctq->bdtq")
    return torch.stack(
        [
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1]),
        ],
        dim=-1,
    )


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(out_channels, in_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(out_channels, in_channels, self.modes1, self.modes2, 2)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # x_ft = torch.rfft(x, 2, normalized=True, onesided=True)
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.view_as_real(x_ft)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            2,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, :] = compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2, :], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, :] = compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2, :], self.weights2
        )
        out_ft = torch.view_as_complex(out_ft)

        # Return to physical space
        # x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SimpleBlock2d(nn.Module):
    def __init__(self, modes):
        super(SimpleBlock2d, self).__init__()

        self.conv1 = SpectralConv2d(1, 16, modes=modes)
        self.conv2 = SpectralConv2d(16, 32, modes=modes)
        self.conv3 = SpectralConv2d(32, 64, modes=modes)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Net2d(nn.Module):
    def __init__(self):
        super(Net2d, self).__init__()

        self.conv = SimpleBlock2d(5)

    def forward(self, x):
        x = self.conv(x)

        return x.squeeze(-1)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, modes=10):
        super(BasicBlock, self).__init__()
        self.conv1 = SpectralConv2d(in_planes, planes, modes=modes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SpectralConv2d(planes, planes, modes=modes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SpectralConv2d(in_planes, self.expansion * planes, modes=modes),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FNO(nn.Module):
    def __init__(
        self, block="BasicBlock", num_blocks=[3, 4, 23, 3], num_classes=10, modes=3
    ):
        super(FNO, self).__init__()
        self.in_planes = 32

        self.conv1 = SpectralConv2d(3, 32, modes=10)
        self.bn1 = nn.BatchNorm2d(32)
        block = eval(block)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1, modes=modes)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=1, modes=modes)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=1, modes=modes)
        self.layer4 = self._make_layer(block, 32, num_blocks[3], stride=1, modes=modes)
        # self.layer5 = self._make_layer(block, 3, num_blocks[3], stride=1, modes=5)
        self.conv2 = SpectralConv2d(32, 3, modes=10)

        # self.linear1 = nn.Linear(32*64*block.expansion, num_classes)
        # self.linear2 = nn.Linear(100, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, modes=10):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, modes))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        # out = F.avg_pool2d(out, 2)
        out = self.layer2(out)
        # out = F.avg_pool2d(out, 2)
        out = self.layer3(out)
        # out = F.avg_pool2d(out, 2)
        out = self.layer4(out)
        out = self.conv2(out)
        # out = F.avg_pool2d(out, 4)
        # out = self.layer5(out)
        # out = out.view(out.size(0), -1)
        # out = self.linear1(out)
        # out = F.relu(out)
        # out = self.linear2(out)
        return out
