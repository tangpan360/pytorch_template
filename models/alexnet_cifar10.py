import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    """
    AlexNet 模型适配 CIFAR-10 数据集
    输入(3, 32, 32) => 输出 num_classes
    由于 CIFAR-10 图像分辨率较低(32x32)，需要调整原始 AlexNet 的一些卷积核参数和全连接层的输入尺寸。
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        # 特征提取部分 (Feature Extractor)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 输入 (3, 32, 32)，输出 (64, 32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 (64, 16, 16)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)  # 输出 (192, 16, 16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 (192, 8, 8)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)  # 输出 (384, 8, 8)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)  # 输出 (256, 8, 8)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # 输出 (256, 8, 8)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 输出 (256, 4, 4)

        # 分类器部分 (Classifier)
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)  # 展平后输入 (256*4*4 = 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

        # Dropout 用于减少过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 特征提取部分
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
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
