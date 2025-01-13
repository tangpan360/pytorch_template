import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class MnistDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None, target_transforme=None):
        """
        初始化数据集。

        参数：
            annotatinos_file (str): CSV 文件路径，包含 (filename, label) 等信息
            img_dir (str): 存放图像的文件夹路径
            transform: 对图像进行转换和增强的函数或 transforms 组合
            target_transform: 对标签进行转换的函数
        """
        self.data_dir = data_dir
        self.img_path = os.listdir(self.data_dir)
        self.img_labels = pd.read_csv(labels_csv)
        self.transform = transform
        self.target_transform = target_transforme

    def __getitem__(self, idx):
        """
        根据索引 idx 获取单个样本。

        返回:
            (image, label) 其中 image 可以是一个 PIL 图像或 Tensor，label 可以是整数或字符串
        """
        img_item_path = os.path.join(self.data_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]

        image = Image.open(img_item_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        """返回整个数据集的样本数量。"""
        return len(self.img_labels)
