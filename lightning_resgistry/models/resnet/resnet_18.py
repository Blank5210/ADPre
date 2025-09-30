# 处理图像尺寸为152*188*152
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

from lightning_resgistry.models.builder import MODELS



class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNetFeature(nn.Module):
    def __init__(self, ResBlock):
        super(ResNetFeature, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool3d(3, 2)
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.layer_norm = nn.LayerNorm(512)
        # self.fc = nn.Linear(512, num_classes)

    # 用来重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # 填充操作
        padding = (1, 0, 1, 0, 1, 0)  # (pad_d1, pad_d2, pad_h1, pad_h2, pad_w1, pad_w2)
        # 应用填充
        out = F.pad(out, padding, mode='constant', value=0)
        out = F.avg_pool3d(out, 3)
        out = out.view(out.size(0), -1)

        return out


@MODELS.register_module("resnet_18")
class Resnet18(nn.Module):
    def __init__(self, num_classes=3, embed_dim=512):
        super(Resnet18, self).__init__()
        self.feature_extractor = ResNetFeature(ResBlock)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = x[:, 0:1, :, :, :]
        mri_features = self.feature_extractor(x)
        out = self.fc(mri_features)
        return out