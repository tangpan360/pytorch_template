# utils/trainer_custom_bert.py
import torch
from tqdm import tqdm


class TrainerCustomBert:
    def __init__(self, model, criterion, optimizer, device, scheduler=None):
        """
        初始化 Trainer 类。

        Args:
            model (torch.nn.Module): 训练的模型。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
            device (torch.device): 训练设备（CPU 或 GPU）。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def train_one_epoch(self, dataloader):
        """
        单轮训练。
        返回：平均损失、准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(dataloader, desc="Training", leave=False, disable=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # 调用 scheduler.step() 每个 batch 之后
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
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
            for batch in tqdm(dataloader, desc="Validation", leave=False, disable=True):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * input_ids.size(0)
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
            for batch in tqdm(dataloader, desc="Testing", leave=False, disable=True):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def get_current_lr(self):
        """
        获取当前学习率。

        Returns:
            float: 当前学习率，如果有调度器，则返回调度器的学习率；否则，从优化器获取学习率。
        """
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        else:
            # 如果没有 scheduler，就直接从 optimizer 中取学习率
            return self.optimizer.param_groups[0]['lr']
