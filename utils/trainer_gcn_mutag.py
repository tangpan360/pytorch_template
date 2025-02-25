# utils/trainer_gcn_mutag.py
import torch

class TrainerGCNMutag:
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

    def train_one_epoch(self, train_loader):
        """
        单轮训练。

        Args:
            train_loader (torch_geometric.data.DataLoader): 训练集数据加载器。

        Returns:
            avg_loss (float): 平均损失。
            accuracy (float): 准确率。
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index, data.batch)  # 模型输出

            loss = self.criterion(out, data.y)  # 计算损失
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs  # 计算总损失
            _, predicted = torch.max(out, dim=1)  # 获取预测结果
            total += data.num_graphs  # 计算图的数量
            correct += (predicted == data.y).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate_one_epoch(self, val_loader):
        """
        单轮验证。

        Args:
            val_loader (torch_geometric.data.DataLoader): 验证集数据加载器。

        Returns:
            avg_loss (float): 平均损失。
            accuracy (float): 准确率。
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)  # 模型输出

                loss = self.criterion(out, data.y)  # 计算损失
                total_loss += loss.item() * data.num_graphs
                _, predicted = torch.max(out, dim=1)  # 获取预测结果
                total += data.num_graphs
                correct += (predicted == data.y).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def test_model(self, test_loader):
        """
        测试集评估。

        Args:
            test_loader (torch_geometric.data.DataLoader): 测试集数据加载器。

        Returns:
            avg_loss (float): 平均损失。
            accuracy (float): 准确率。
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)  # 模型输出

                loss = self.criterion(out, data.y)  # 计算损失
                total_loss += loss.item() * data.num_graphs
                _, predicted = torch.max(out, dim=1)  # 获取预测结果
                total += data.num_graphs
                correct += (predicted == data.y).sum().item()

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
