import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNModel(nn.Module):
    """
    基于图卷积网络 (GCN) 的模型，输入为图的节点特征和图的边。
    """

    def __init__(self, num_features, num_classes, dropout_prob=0.5):
        super(GCNModel, self).__init__()

        # 图卷积层：输入节点特征的维度 -> 输出节点特征的维度
        self.conv1 = GCNConv(num_features, 16)  # 第一个GCN层
        self.conv2 = GCNConv(16, num_classes)  # 第二个GCN层，用于输出类别
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, edge_index):
        """
        前向传播，输入为图数据（data）：
        - data.x: 节点特征
        - data.edge_index: 图的边索引
        """
        # 第一个图卷积层
        x = self.conv1(x, edge_index)  # 图卷积操作
        x = F.relu(x)  # 激活函数（ReLU）
        x = self.dropout(x)

        # 第二个图卷积层（不使用激活函数，输出类别）
        x = self.conv2(x, edge_index)

        return x
