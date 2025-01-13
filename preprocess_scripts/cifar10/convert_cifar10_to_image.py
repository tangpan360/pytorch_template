import numpy as np
import cv2
import pickle
import os
from tqdm import tqdm

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_images(data, labels, label_names, save_dir, start_index=0):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 使用 tqdm 包装 enumerate 以显示进度条
    for i, (image_data, label) in tqdm(enumerate(zip(data, labels), start=start_index), total=len(data), desc=f"Saving images to {save_dir}"):
        # CIFAR-10图像数据是按照 [32, 32, 3] 的形状存储的，但数据是一维数组
        # 首先将其转换为 [32, 32, 3] 形状的数组
        image = image_data.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)  # 使用 uint8 类型
        # OpenCV 默认使用 BGR 格式，所以我们需要将图像数据从 RGB 转换为 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filename = f"{label_names[label]}_NO.{i}.png"  # 使用标签和索引构建文件名
        cv2.imwrite(os.path.join(save_dir, filename), image)  # 保存图像
    print(f"\n{len(data)} images saved to {save_dir}")

# 定义CIFAR-10的所有类别名称
label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 定义CIFAR-10数据集的文件路径
base_path = "../../dataset/cifar10/raw/cifar-10-batches-py/"
train_batch_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
test_batch_files = ["test_batch"]

# 合并所有训练集数据到 train_data 文件夹中
train_save_dir = "../../dataset/cifar10/processed/train_data"
all_train_data = []
all_train_labels = []

for batch_file in tqdm(train_batch_files, desc="Loading training batches"):
    data_batch = unpickle(os.path.join(base_path, batch_file))
    all_train_data.append(data_batch[b'data'])
    all_train_labels.extend(data_batch[b'labels'])

# 将所有训练集数据拼接成一个大的 NumPy 数组
all_train_data = np.concatenate(all_train_data)

# 计算训练集的起始索引，以确保编号连续且不重复
start_index = 0

# 保存所有训练集图像到 train_data 文件夹中
save_images(all_train_data, all_train_labels, label_name, train_save_dir, start_index=start_index)

# 更新起始索引以便测试集编号不会与训练集冲突
start_index += len(all_train_data)

# 将测试集数据保存到 test_data 文件夹中
test_save_dir = "../../dataset/cifar10/processed/test_data"
for batch_file in tqdm(test_batch_files, desc="Loading test batches"):
    data_batch = unpickle(os.path.join(base_path, batch_file))
    cifar_data = data_batch[b'data']
    cifar_labels = data_batch[b'labels']
    save_images(cifar_data, cifar_labels, label_name, test_save_dir, start_index=start_index)
    start_index += len(cifar_data)

print("All images have been saved.")