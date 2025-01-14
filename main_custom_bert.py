# main.py
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import json

from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models import CustomBertForClassification

# 导入你自己的数据集类
from dataset_class import AGNewsDataset

# 导入封装好的类和函数
from utils import TrainerCustomBert
from utils.early_stopping import EarlyStopping
from utils.time_utils import format_time
from utils.seed_utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple AG News dataset.")
    # 常用超参数示例，实际使用中可按需调整
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_classes", type=int, default=4, help="类别数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批量大小")

    # scheduler 相关参数
    parser.add_argument("--use_scheduler", action="store_true", default=True, help="是否使用学习率调度器")
    parser.add_argument("--warmup_steps", type=int, default=6000, help="学习率预热步数")

    # 针对 AGNews 的数据路径配置
    parser.add_argument("--train_path", type=str, default="./data/ag_news/processed/train.parquet", help="训练集数据路径")
    parser.add_argument("--val_path", type=str, default="./data/ag_news/processed/val.parquet", help="验证集数据路径")
    parser.add_argument("--test_path", type=str, default="./data/ag_news/processed/test.parquet", help="测试集数据路径")

    # 将 tokenizer 和 model 名字或路径作为参数传入
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-tiny", help="HuggingFace Tokenizer 名称或路径")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny", help="HuggingFace 预训练模型名称或路径")

    # 其它可选参数
    parser.add_argument("--max_length", type=int, default=128, help="文本最大长度")
    parser.add_argument("--log_dir", type=str, default="./logs/ag_news_custom_bert/", help="日志文件夹路径")

    # EarlyStopping相关
    parser.add_argument("--patience", type=int, default=5, help="早停的等待轮数")
    parser.add_argument("--delta", type=float, default=0.0, help="判断改善的阈值")
    parser.add_argument("--early_stop_metric", type=str, default="loss", choices=["loss", "acc"],
                        help="选择使用验证集损失('loss')或准确率('acc')进行早停")
    parser.add_argument("--early_stop_verbose", action="store_false", default=True, help="是否启用早停机制中的详细输出。"
                        "如果设置为 True，则在早停检查点和模型保存时打印提示信息；如果设置为 False，则不打印这些信息。默认值为 True。")
    parser.add_argument("--save_path", type=str, default="./checkpoints/ag_news_custom_bert/best_model.pth",
                        help="最优模型权重保存路径（含文件名）")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)

    # 构建数据集
    train_dataset = AGNewsDataset(
        data_path=args.train_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    val_dataset = AGNewsDataset(
        data_path=args.val_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_dataset = AGNewsDataset(
        data_path=args.test_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # 构建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomBertForClassification(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout_prob=0.3
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    if args.use_scheduler:
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_steps
        )
    else:
        scheduler = None

    # 初始化 Trainer
    trainer = TrainerCustomBert(model, criterion, optimizer, device, scheduler=scheduler)

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
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "training_metrics.jsonl")  # 每个 epoch 一行

    # 如果文件存在，就删除
    if os.path.exists(log_path):
        os.remove(log_path)

    # 记录训练开始时间
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # ---------- Training ---------- #
        train_loss, train_acc = trainer.train_one_epoch(train_loader)

        # 获取当前学习率
        current_lr = trainer.get_current_lr()

        # ---------- Validation ---------- #
        val_loss, val_acc = trainer.validate_one_epoch(val_loader)

        # 输出信息
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
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
