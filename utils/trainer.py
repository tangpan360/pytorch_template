# utils/trainer.py
import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, device, scheduler=None, use_amp=False, gradient_clip_norm=None, gradient_clip_value=None):
        """
        初始化 Trainer 类。

        Args:
            model (torch.nn.Module): 训练的模型。
            criterion (torch.nn.Module): 损失函数。
            optimizer (torch.optim.Optimizer): 优化器。
            device (torch.device): 训练设备（CPU 或 GPU）。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。
            use_amp (bool): 是否启用混合精度训练。
            gradient_clip_norm (float, optional): 梯度裁剪的范数阈值。
            gradient_clip_value (float, optional): 梯度裁剪的值阈值。
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value

    def train_one_epoch(self, dataloader):
        """
        单轮训练。
        返回：平均损失、准确率
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc="Training", leave=True, disable=False):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.use_amp:
                # 使用 GradScaler 进行反向传播和优化
                self.scaler.scale(loss).backward()

                # 基于范数的裁剪
                if self.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                # 基于值的裁剪
                elif self.gradient_clip_value is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # 基于范数的裁剪
                if self.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                
                # 基于值的裁剪
                elif self.gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip_value)

                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 在 epoch 结束时更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

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
            for images, labels in tqdm(dataloader, desc="Validation", leave=True, disable=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 混合精度推理
                with torch.cuda.amp.autocast(enabled=self.use_amp):
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
            for images, labels in tqdm(dataloader, desc="Testing", leave=True, disable=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # 混合精度推理
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
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
