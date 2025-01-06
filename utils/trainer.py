# utils/trainer.py
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        """
        初始化 Trainer 类。

        Args:
            model (torch.nn.Module): 训练的模型。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
            device (torch.device): 训练设备（CPU 或 GPU）。
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        """
        单轮训练。
        返回：平均损失、准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc="Training", leave=False, disable=True):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate_one_epoch(self, dataloader):
        """
        单轮验证。
        返回：平均损失、准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation", leave=False, disable=True):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def test_model(self, dataloader):
        """
        测试集评估。
        返回：平均损失、准确率
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Testing", leave=False, disable=True):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
