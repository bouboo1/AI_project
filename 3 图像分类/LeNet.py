import torch.nn as nn
from torch.nn import functional as F
class LeNet(nn.Module):
    def __init__(self, dropout):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            # 卷积层
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            # 池化层
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            # 卷积层
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Sigmoid(), 
            nn.BatchNorm2d(16),
            # 池化层
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        )

        self.classifier = nn.Sequential(
            # 全连接层
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            # 输出
            nn.Linear(84, 10),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = F.interpolate(x, size=32, mode='bilinear', align_corners=False)
        feature = self.feature_extractor(x)
        feature = feature.reshape(x.size(0), -1)
        output = self.classifier(feature)
        return output