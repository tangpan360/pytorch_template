import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv


class GCNMutag(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, dropout_prob=0.5):
        super(GCNMutag, self).__init__()
        # GCN 卷积层
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # 分类层
        self.fc = nn.Linear(hidden_channels, num_classes)

        # 其他组件
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, batch):  # 添加 batch 参数
        # GCN 特征提取
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)  # 最后一层不接激活函数

        # 图级池化
        x = pyg_nn.global_mean_pool(x, batch)  # 使用传入的 batch

        # 分类输出
        x = self.dropout(x)
        x = self.fc(x)

        return x
