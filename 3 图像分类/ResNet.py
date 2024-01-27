'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-08 14:16:17
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-08 14:16:19
FilePath: \10215501433 仲韦萱 实验三\ResNet.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from torch.nn import functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample):
        super(ResNetBlock, self).__init__()
        self.down_sample = down_sample

        self.DownSample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2)),
            nn.BatchNorm2d(out_channels)
        )

        if down_sample:
            self.Conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.Conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.Conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.Conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = self.Conv1(x)
        output = self.Conv2(output)

        if self.down_sample:
            return F.relu(self.DownSample(x) + output)
        else:
            return F.relu(x + output)

class ResNet(nn.Module):
    def __init__(self, dropout):
        super(ResNet, self).__init__()
        self.Conv = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.BN = nn.BatchNorm2d(64)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Layer1 = nn.Sequential(
            ResNetBlock(in_channels=64, out_channels=64, down_sample=False),
            ResNetBlock(in_channels=64, out_channels=64, down_sample=False)
        )

        self.Layer2 = nn.Sequential(
            ResNetBlock(in_channels=64, out_channels=128, down_sample=True),
            ResNetBlock(in_channels=128, out_channels=128, down_sample=False)
        )

        self.Layer3 = nn.Sequential(
            ResNetBlock(in_channels=128, out_channels=256, down_sample=True),
            ResNetBlock(in_channels=256, out_channels=256, down_sample=False)
        )

        self.Layer4 = nn.Sequential(
            ResNetBlock(in_channels=256, out_channels=512, down_sample=True),
            ResNetBlock(in_channels=512, out_channels=512, down_sample=False)
        )

        self.AvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.FC = nn.Linear(512, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.Conv(x)
        output = self.BN(output)
        output = self.MaxPool(output)
        output = self.Layer1(output)
        output = self.Layer2(output)
        output = self.Layer3(output)
        output = self.Layer4(output)
        output = self.AvgPool(output)
        output = output.view(output.size(0), -1)
        output = self.FC(output)
        output = self.dropout(output)
        return output