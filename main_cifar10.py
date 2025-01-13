# main.py
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import json  # <-- 用于写JSON

from torch.utils.data import DataLoader
from torchvision import transforms

# 导入你自己的数据集类
from datasets import Cifar10Dataset

# 导入封装好的类和函数
from utils.trainer import Trainer
from utils.early_stopping import EarlyStopping
from utils.time_utils import format_time
from utils.seed_utils import set_seed

# 你的模型
from models import VGGNetCifar10


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple LeNet on custom CIFAR-10 data.")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_classes", type=int, default=10, help="类别数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--train_dir", type=str, default="./data/cifar10/processed/train_data", help="训练集数据位置")
    parser.add_argument("--train_labels", type=str, default="./data/cifar10/processed/train_annotations.csv", help="训练集数据标签csv")
    parser.add_argument("--val_dir", type=str, default="./data/cifar10/processed/val_data", help="验证集数据位置")
    parser.add_argument("--val_labels", type=str, default="./data/cifar10/processed/val_annotations.csv", help="验证集数据标签csv")
    parser.add_argument("--test_dir", type=str, default="./data/cifar10/processed/test_data", help="测试集数据位置")
    parser.add_argument("--test_labels", type=str, default="./data/cifar10/processed/test_annotations.csv", help="测试集数据标签csv")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")

    parser.add_argument("--patience", type=int, default=10, help="早停的等待轮数")
    parser.add_argument("--delta", type=float, default=0.0, help="判断改善的阈值")
    parser.add_argument("--early_stop_metric", type=str, default="loss", choices=["loss", "acc"],
                        help="选择使用验证集损失('loss')或准确率('acc')进行早停")
    parser.add_argument("--early_stop_verbose", type=bool, default=True, help="是否启用早停机制中的详细输出。"
                  "如果设置为 True，则在早停检查点和模型保存时打印提示信息；如果设置为 False，则不打印这些信息。默认值为 True。")

    # 新增的权重保存路径参数
    parser.add_argument("--save_path", type=str, default="./checkpoints/cifar10/best_model.pth",
                        help="最优模型权重保存路径（含文件名）")

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
    train_dataset = Cifar10Dataset(
        data_dir=args.train_dir,
        labels_csv=args.train_labels,
        transform=train_transform,
    )
    val_dataset = Cifar10Dataset(
        data_dir=args.val_dir,
        labels_csv=args.val_labels,
        transform=val_transform,
    )
    test_dataset = Cifar10Dataset(
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
    model = VGGNetCifar10(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 初始化 Trainer
    trainer = Trainer(model, criterion, optimizer, device)

    # EarlyStopping 监控指标
    if args.early_stop_metric == "loss":
        monitor = 'val_loss'
    else:
        monitor = 'val_acc'

    # 在使用 save_path 之前，先创建对应目录（若不存在）
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    early_stopper = EarlyStopping(
        patience=args.patience,
        verbose=args.early_stop_verbose,
        delta=args.delta,
        save_path=args.save_path,
        monitor=monitor
    )

    # ========== 1) 创建日志文件夹 ========== #
    os.makedirs("logs/cifar10/", exist_ok=True)
    log_path = "logs/cifar10/training_metrics.jsonl"  # 每个 epoch 一行

    # 如果文件存在，就删除
    if os.path.exists(log_path):
        os.remove(log_path)

    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # ---------- Training ---------- #
        train_loss, train_acc = trainer.train_one_epoch(train_loader)

        # ---------- Validation ---------- #
        val_loss, val_acc = trainer.validate_one_epoch(val_loader)

        # 输出信息
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_remaining_time = avg_time_per_epoch * remaining_epochs

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
              f"| Time elapsed: {format_time(elapsed_time)} "
              f"| Estimated remaining: {format_time(estimated_remaining_time)}")

        # ========== 2) 写入 JSON Lines：一行一个 epoch ========== #
        log_dict = {
            "epoch": epoch + 1,
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
