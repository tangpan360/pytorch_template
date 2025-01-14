import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class CustomBertForClassification(nn.Module):
    def __init__(self, model_name: str, num_classes: int=4, dropout_prob: float=0.3):
        """
        初始化自定义 BERT 模型，用于文本分类。

        Args:
            model_name (str): 预训练的 BERT 模型名称或路径。
            num_classes (int): 分类类别数，默认值为4。
            dropout_prob (float): Dropout 的概率，默认值为0。3.
        """
        super(CustomBertForClassification, self).__init__()
        self.num_classes = num_classes

        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(model_name)

        # 获取 BERT 模型的隐藏层大小
        hidden_size = self.bert.config.hidden_size

        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob)

        # 两层线性层 + ReLU 激活
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)  # 第一层线性层
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)  # 第二层线性层

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化分类部分的权重。
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, input_ids, attention_mask):
        """
        前向传播。

        Args:
            input_ids (torch.Tensor): 输入的 token IDs，形状为 (batch_size, sequence_length)。
            attention_mask (torch.Tensor): 注意力掩码，形状为 (batch_size, sequence_length)。

        Returns:
            torch.Tensor: 分类 logits，形状为 (batch_size, num_classes)。
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 使用 [CLS] 标记的输出作为句子的表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # 应用 Dropout
        cls_output = self.dropout(cls_output)

        # 分类头
        hidden = self.fc1(cls_output)  # (batch_size, hidden_size // 2)
        hidden = self.relu(hidden)  # (batch_size, hidden // 2)
        logits = self.fc2(hidden)  # (batch_size, num_classes)

        return logits
