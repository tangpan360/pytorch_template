# main.py
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import json
import random
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR  # 添加学习率调度器

# 导入你自己的数据集类
from dataset_class import MnistDataset

# 导入封装好的类和函数
from utils.trainer import Trainer
from utils.early_stopping import EarlyStopping
from utils.time_utils import format_time
from utils.seed_utils import set_seed

# 你的模型
from models import VGGNetMnist


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple VGGNet on MNIST data.")
    
    # 常用超参数，实际使用时可根据需要调整
    parser.add_argument("--seed", type=int, default=42, help="随机种子，用于确保结果的可重复性")
    parser.add_argument("--num_classes", type=int, default=10, help="分类问题的类别数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率，控制参数更新的步长")
    parser.add_argument("--epochs", type=int, default=50, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="每个批次的数据量")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 正则化系数，防止模型过拟合")

    # 学习率调度器相关参数
    parser.add_argument("--use_scheduler", action="store_true", default=False, help="是否启用学习率调度器")
    parser.add_argument("--scheduler_step_size", type=int, default=1, help="学习率衰减的步长")
    parser.add_argument("--scheduler_gamma", type=float, default=0.9, help="学习率衰减的因子")

    # 数据路径配置
    parser.add_argument("--train_dir", type=str, default="./data/mnist/processed/train_data", help="训练集数据位置")
    parser.add_argument("--train_labels", type=str, default="./data/mnist/processed/train_annotations.csv", help="训练集数据标签csv")
    parser.add_argument("--val_dir", type=str, default="./data/mnist/processed/val_data", help="验证集数据位置")
    parser.add_argument("--val_labels", type=str, default="./data/mnist/processed/val_annotations.csv", help="验证集数据标签csv")
    parser.add_argument("--test_dir", type=str, default="./data/mnist/processed/test_data", help="测试集数据位置")
    parser.add_argument("--test_labels", type=str, default="./data/mnist/processed/test_annotations.csv", help="测试集数据标签csv")
    parser.add_argument("--log_dir", type=str, default="./logs/mnist/", help="日志文件夹路径")

    # 早停相关参数
    parser.add_argument("--patience", type=int, default=10, help="早停的等待轮数")
    parser.add_argument("--delta", type=float, default=0.0, help="判断改善的阈值")
    parser.add_argument("--early_stop_metric", type=str, default="loss", choices=["loss", "acc"],
                        help="选择使用验证集损失('loss')或准确率('acc')进行早停")
    parser.add_argument("--early_stop_verbose", action="store_true", default=True, 
                        help="是否启用早停机制中的详细输出。如果设置为 True，则在早停检查点和模型保存时打印提示信息；如果设置为 False，则不打印这些信息。")

    # 模型保存路径参数
    parser.add_argument("--save_path", type=str, default="./checkpoints/mnist/best_model.pth",
                        help="最优模型权重保存路径（含文件名）")

    # Checkpoint 保存与加载相关参数
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="是否从保存的 checkpoint 恢复训练，False 表示不启用")
    parser.add_argument("--save_checkpoints", action="store_true", default=False, help="是否保存 checkpoint 以便后续恢复训练，False 表示不启用")
    parser.add_argument("--checkpoint_freq", type=int, default=2, help="每经过多少个 epoch 保存一次 checkpoint")

    # 混合精度训练选项
    parser.add_argument("--use_amp", action="store_true", default=False, help="是否启用混合精度训练，False 表示不启用")

    # 梯度裁剪相关参数
    parser.add_argument("--gradient_clip_norm", type=float, default=None, help="基于范数的梯度裁剪最大值，None 表示不启用（常用值：0.5-5.0）")
    parser.add_argument("--gradient_clip_value", type=float, default=None, help="基于值的梯度裁剪最大值，None 表示不启用（常用值：0.1-1.0）")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 构建数据集
    train_dataset = MnistDataset(
        data_dir=args.train_dir,
        labels_csv=args.train_labels,
        transform=train_transform,
    )
    val_dataset = MnistDataset(
        data_dir=args.val_dir,
        labels_csv=args.val_labels,
        transform=val_transform,
    )
    test_dataset = MnistDataset(
        data_dir=args.test_dir,
        labels_csv=args.test_labels,
        transform=test_transform,
    )

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGGNetMnist(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = None
    if args.use_scheduler:
        # 使用StepLR，在指定steps后乘以gamma进行衰减
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # 初始化 Trainer
    trainer = Trainer(model, criterion, optimizer, device, scheduler=scheduler, 
                     use_amp=args.use_amp, gradient_clip_norm=args.gradient_clip_norm, 
                     gradient_clip_value=args.gradient_clip_value)

    # EarlyStopping 监控指标
    if args.early_stop_metric == "loss":
        monitor = 'val_loss'
    else:
        monitor = 'val_acc'

    # 在使用 save_path 之前，先创建对应目录（若不存在）
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 加载最新的 checkpoint，如果需要恢复
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_") and f.endswith(".pth")]
        if checkpoint_files:
            print("检测到从断点恢复训练，正在恢复训练状态...\n")
            # 获取最新的 checkpoint 文件
            latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[1].split('.')[0]))
            checkpoint_path = os.path.join(save_dir, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path)

            # 恢复模型和优化器状态
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型状态已恢复")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("优化器状态已恢复")

            # 恢复学习率调度器状态
            if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("学习率调度器状态已恢复")

            # 恢复随机状态
            if 'rng_state' in checkpoint:
                random.setstate(checkpoint['rng_state']['python'])
                np.random.set_state(checkpoint['rng_state']['numpy'])
                torch.set_rng_state(checkpoint['rng_state']['torch'])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
                print("随机状态已恢复")

            # 恢复混合精度训练 scaler 状态
            if 'scaler_state_dict' in checkpoint and hasattr(trainer, 'scaler') and trainer.scaler is not None:
                trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print("混合精度 scaler 状态已恢复")

            # 恢复 epoch 计数
            start_epoch = checkpoint['epoch'] + 1  # 加 1 以便从下一个 epoch 开始
            print(f"从 {checkpoint_path} 加载 checkpoint，从第 {start_epoch} 个 epoch 开始")
        else:
            print("未找到 checkpoint，从头开始训练")

    early_stopper = EarlyStopping(
        patience=args.patience,
        verbose=args.early_stop_verbose,
        delta=args.delta,
        save_path=args.save_path,
        monitor=monitor
    )

    # ========== 1) 创建日志文件夹 ========== #
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "training_metrics.jsonl")  # 每个 epoch 一行

    # 如果是从头开始训练，则删除已有日志文件；如果恢复训练，则保留原日志，继续追加
    if start_epoch == 0 and os.path.exists(log_path):
        os.remove(log_path)

    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # ---------- Training ---------- #
        train_loss, train_acc = trainer.train_one_epoch(train_loader)

        # 获取当前学习率
        current_lr = trainer.get_current_lr() if hasattr(trainer, 'get_current_lr') else optimizer.param_groups[0]['lr']

        # ---------- Validation ---------- #
        val_loss, val_acc = trainer.validate_one_epoch(val_loader)

        # 输出信息
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1 - start_epoch)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_remaining_time = avg_time_per_epoch * remaining_epochs

        print(f"  LR: {current_lr:.8f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
              f"| Time elapsed: {format_time(elapsed_time)} "
              f"| Estimated remaining: {format_time(estimated_remaining_time)}")

        # ========== 2) 写入 JSON Lines：一行一个 epoch ========== #
        log_dict = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }

        # append模式打开文件，一行写一个JSON
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + "\n")

        # ---------- EarlyStopping ---------- #
        if monitor == 'val_loss':
            early_stopper(val_loss, trainer.model)
        else:
            early_stopper(val_acc, trainer.model)

        if early_stopper.early_stop:
            print("Early stopping triggered. Stop training.")
            break
            
        # ========== 3) 保存 checkpoint ========== #
        if args.save_checkpoints and (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_save_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}.pth")
            # 保存模型、优化器、学习率调度器、随机状态及混合精度训练 scaler
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                # 保存随机状态
                'rng_state': {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                },
                # 只有在启用混合精度训练时才保存 scaler 状态
                'scaler_state_dict': trainer.scaler.state_dict() if args.use_amp and hasattr(trainer, 'scaler') else None
            }, checkpoint_save_path)
            print(f"Checkpoint 已保存至 {checkpoint_save_path}")

    # ---------- 测试集评估最优模型 ---------- #
    checkpoint_path = early_stopper.save_path
    try:
        trainer.model.load_state_dict(torch.load(checkpoint_path))
        print(f"\nLoaded the best model weights from {checkpoint_path} for testing.")
    except FileNotFoundError:
        print("Warning: best model weights not found, using current model.")

    print("\nEvaluating on the test set...")
    test_loss, test_acc = trainer.test_model(test_loader)
    print(f"  Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
