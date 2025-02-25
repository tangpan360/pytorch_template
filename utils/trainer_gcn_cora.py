# utils/trainer_gcn.py
import torch


class TrainerGCN:
    def __init__(self, model, criterion, optimizer, device, scheduler=None):
        """
        初始化 Trainer 类。

        Args:
            model (torch.nn.Module): 训练的模型。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
            device (torch.device): 训练设备（CPU 或 GPU）。
            scheduler (Optional[torch.optim.lr_scheduler]): 学习率调度器（可选）
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_one_epoch(self, data, train_mask):
        """
        单轮训练。

        Args:
            data (torch_geometric.data.Data): 图数据对象。
            train_mask (torch.Tensor): 训练集掩码。

        Returns:
            avg_loss (float): 平均损失。
            accuracy (float): 准确率。
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # 使用训练集掩码进行训练
        self.optimizer.zero_grad()
        data = data.to(self.device)
        out = self.model(data.x, data.edge_index)  # 模型输出

        loss = self.criterion(out[train_mask], data.y[train_mask])  # 计算损失
        num_selected_nodes = train_mask.sum().item()
        loss.backward()
        self.optimizer.step()

        total_loss += loss.item() * num_selected_nodes
        _, predicted = torch.max(out[train_mask], dim=1)  # 获取预测结果
        total += num_selected_nodes  # 计算训练集中的节点数
        correct += (predicted == data.y[train_mask]).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate_one_epoch(self, data, val_mask):
        """
        单轮验证。

        Args:
            data (torch_geometric.data.Data): 图数据对象。
            val_mask (torch.Tensor): 验证集掩码。

        Returns:
            avg_loss (float): 平均损失。
            accuracy (float): 准确率。
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)  # 模型输出

            loss = self.criterion(out[val_mask], data.y[val_mask])  # 计算损失
            num_selected_nodes = val_mask.sum().item()
            total_loss += loss.item() * num_selected_nodes
            _, predicted = torch.max(out[val_mask], dim=1)  # 获取预测结果
            total += num_selected_nodes  # 计算验证集中的节点数
            correct += (predicted == data.y[val_mask]).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def test_model(self, data, test_mask):
        """
        测试集评估。

        Args:
            data (torch_geometric.data.Data): 图数据对象。
            test_mask (torch.Tensor): 测试集掩码。

        Returns:
            avg_loss (float): 平均损失。
            accuracy (float): 准确率。
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index)  # 模型输出

            loss = self.criterion(out[test_mask], data.y[test_mask])  # 计算损失
            num_selected_nodes = test_mask.sum().item()
            total_loss += loss.item() * num_selected_nodes
            _, predicted = torch.max(out[test_mask], dim=1)  # 获取预测结果
            total += num_selected_nodes  # 计算测试集中的节点数
            correct += (predicted == data.y[test_mask]).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def get_current_lr(self):
        """
        获取当前学习率。

        Returns:
            current_lr (float): 当前学习率。
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def step_scheduler(self):
        """
        更新学习率调度器。
        """
        if self.scheduler:
            self.scheduler.step()
