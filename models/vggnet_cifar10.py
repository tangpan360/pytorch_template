import torch.nn as nn
import torch.nn.functional as F


class VGGNetCifar10(nn.Module):
    """
    简易 VGGNet: 输入(3, 32, 32) => 输出 num_classes
    适配 CIFAR-10 数据集的小型 VGGNet。卷积核使用 3x3，所有池化层的步幅均为 2，特征图尺寸逐步减小。
    """

    def __init__(self, num_classes=10):
        super(VGGNetCifar10, self).__init__()

        # 特征提取部分 (Feature Extractor)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 输入 (3, 32, 32) => 输出 (64, 32, 32)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # 输出 (64, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 (64, 16, 16)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 输出 (128, 16, 16)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 输出 (128, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 (128, 8, 8)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 输出 (256, 8, 8)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 输出 (256, 8, 8)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 输出 (256, 8, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 (256, 4, 4)

        # 分类器部分 (Classifier)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 特征提取部分
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)

        # 展平张量为 1D
        x = x.view(-1, 256 * 4 * 4)

        # 分类器部分
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
