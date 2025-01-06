# **深度学习基础框架模板 (Pytorch Template)**

## 项目简介

项目链接：[https://github.com/tangpan360/pytorch_template](https://github.com/tangpan360/pytorch_template)

本项目是一个基于 PyTorch 的深度学习基础框架，旨在帮助用户快速实现自己的训练模型。通过替换数据集和数据预处理等模块，用户可以专注于模型开发和实验，而无需花费大量时间在基础功能的实现上，比如：

- 可视化（loss 和 acc 变化曲线）  
- 模型早停机制（Early Stopping）  
- 随机种子设置  
- 数据加载和预处理  
- 训练日志记录  

框架结构清晰、模块化设计，便于扩展和复用，同时包含了一些常用的深度学习工具和方法。既适合新手快速上手，也适合高级用户构建自己的实验框架。

---
## 运行结果展示
1. 终端输出：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8596c6f9f9794ce99a17595760880e56.png)
2. 损失和准确率可视化实时更新：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/46cf03e0fb9642f8a3199aad0e8b4c7d.png)
或者：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/23389e3297a543bd9ebb807aed7fb45b.png)
3. 参数配置： ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a239cef08a4c4186a88cb1d89c54e137.png)

## 文件和目录结构说明

```
pytorch_template
├── checkpoints
│   └── best_model.pth                 # 模型训练后的最佳权重文件
├── dataset
│   ├── processed
│   │   └── cifar-10
│   │       ├── test_data/             # 处理后的测试集数据
│   │       ├── train_data/            # 处理后的训练集数据
│   │       ├── val_data/              # 处理后的验证集数据
│   │       ├── full_train_annotations.csv  # 完整训练集的标签文件
│   │       ├── test_annotations.csv   # 测试集的标签文件
│   │       ├── train_annotations.csv  # 训练集的标签文件
│   │       └── val_annotations.csv    # 验证集的标签文件
│   └── raw
│       ├── cifar-10-batches-py        # CIFAR-10 原始数据解压后的目录
│       │   ├── batches.meta
│       │   ├── data_batch_1
│       │   ├── data_batch_2
│       │   ├── data_batch_3
│       │   ├── data_batch_4
│       │   ├── data_batch_5
│       │   ├── readme.html
│       │   └── test_batch
│       └── cifar-10-python.tar.gz     # CIFAR-10 数据的原始压缩包
├── logs
│   ├── train_log.txt                  # 训练日志
│   └── training_metrics.jsonl         # 训练过程中的指标记录（JSON 行格式）
├── models
│   ├── AlexNet.py                     # AlexNet 模型定义
│   ├── LeNet.py                       # LeNet 模型定义
│   └── VGGNet.py                      # VGGNet 模型定义
├── preprocess_scripts
│   ├── convert_cifar10_to_image.py    # 脚本：将 CIFAR-10 数据转换为图片格式
│   ├── download_cifar10.py            # 脚本：下载 CIFAR-10 数据集
│   └── generate_annotations.ipynb     # 脚本：生成训练/验证/测试集的标签文件
├── utils
│   ├── __init__.py                    # 工具模块的初始化文件
│   ├── cifar10_dataset.py             # 自定义 CIFAR-10 数据集加载工具
│   ├── early_stopping.py              # 早停机制的实现
│   ├── seed_utils.py                  # 随机种子设置工具
│   ├── time_utils.py                  # 时间处理工具
│   └── trainer.py                     # 训练流程封装工具
├── visualization
│   └── visualization_loss_acc.ipynb   # 可视化脚本：展示 loss 和 acc 曲线
├── requirements.txt
├── README.md
├── main.py                            # 主入口：训练脚本
└── run_main.sh                        # 运行训练的 Shell 脚本
```

---

## 功能模块详解

### 1. 数据相关

- **`dataset/raw`**：存放原始数据集文件（如 CIFAR-10 的压缩包）。  
- **`dataset/processed`**：存放预处理后的数据集文件，包括训练集、测试集、验证集和对应的标签文件。

**预处理脚本**：

- `preprocess_scripts/download_cifar10.py`：下载并解压 CIFAR-10 数据集。  
- `preprocess_scripts/convert_cifar10_to_image.py`：将 CIFAR-10 数据集转换为图片格式。  
- `preprocess_scripts/generate_annotations.ipynb`：生成训练/验证/测试集的标签文件。

### 2. 模型相关

- **`models`**：存放常见深度学习模型的定义文件。
  - `AlexNet.py`：AlexNet 模型的实现。
  - `LeNet.py`：LeNet 模型的实现。
  - `VGGNet.py`：VGGNet 模型的实现。

您可以在该目录下添加或修改自己的模型文件。

### 3. 工具函数

- **`utils`**：存放训练与辅助功能的实现，包括：
  - `cifar10_dataset.py`：自定义数据集类，用于加载和处理 CIFAR-10 数据。
  - `early_stopping.py`：实现 Early Stopping，用于防止过拟合。
  - `seed_utils.py`：随机种子设置工具，确保实验结果可重复。
  - `time_utils.py`：时间工具，用于规范化地输出或计算时间。
  - `trainer.py`：封装训练流程的工具，用于简化训练代码。

### 4. 可视化

- **`visualization`**：存放可视化相关脚本。
  - `visualization_loss_acc.ipynb`：通过 Jupyter Notebook 可视化训练过程中的 loss 和 acc 曲线。

### 5. 训练和日志

- **`logs`**：存放训练过程中生成的日志和指标记录文件。
  - `train_log.txt`：记录训练过程中的日志信息（损失、精度等）。
  - `training_metrics.jsonl`：以 JSON 行格式保存的训练指标记录。
- **`checkpoints`**：存放模型训练过程中保存的权重文件（如最佳模型 `best_model.pth`）。

### 6. 主程序

- **`main.py`**：框架主入口，负责整体训练流程。您可以根据需要修改或扩展该文件。  
- **`run_main.sh`**：Shell 脚本，用于一键运行 `main.py`。

---

## 使用方法

### 1. 克隆项目

```bash
git clone https://github.com/tangpan360/pytorch_template.git
cd pytorch_template
```

### 2. 创建并激活 Python 3.9 虚拟环境

```bash
conda create -n pytorch python=3.9
conda activate pytorch
```

### 3. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4. 安装 Jupyter 及相关依赖

如果需要在 Jupyter Notebook 中执行相关脚本，可以安装：

```bash
conda install jupyter
```
若需在 Notebook 中管理多个 conda 环境，可安装：
```bash
conda install nb_conda_kernels
```

---

### 2. 数据准备

1. 下载并解压 CIFAR-10 数据集：
   ```bash
   cd preprocess_scripts
   python download_cifar10.py
   ```
2. 将 CIFAR-10 数据集转换为图片格式：
   ```bash
   python convert_cifar10_to_image.py
   ```
3. 生成训练/验证/测试集的标签文件：
   ```bash
   # 打开并执行 generate_annotations.ipynb
   ```

---

### 3. 开始训练
进入pytorch_template文件夹根目录，直接运行主程序：
```bash
python main.py
```
或通过 Shell 脚本运行：
```bash
bash run_main.sh
```

---

### 4. 可视化结果

在 Jupyter Notebook 中查看训练过程的 Loss 和 Accuracy 曲线：
```bash
# 进入 visualization 文件夹，打开并执行 visualization_loss_acc.ipynb
```

---

## 快速替换自己的数据集

1. 将自有数据放入 `dataset/raw` 目录，并根据情况修改预处理脚本。  
2. 替换 `cifar10_dataset.py` 中的数据加载逻辑。  
3. 调整 `main.py` 中的训练和验证流程，使其适配新的数据集。

---

## TODO

- [ ] 增加更多预训练模型支持（如 ResNet、Transformer 等）。  
- [ ] 支持多 GPU 训练。  
- [ ] 增加更多数据增强功能。  
- [ ] 支持 TensorBoard 可视化。  

---

## 参考与鸣谢

- [PyTorch 官方文档](https://pytorch.org/docs/)
- CIFAR-10 数据集来源：[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
