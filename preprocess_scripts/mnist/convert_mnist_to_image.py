import os
import numpy as np
from PIL import Image

def read_idx(file_path):
    """
    读取 IDX 文件格式
    :param file_path: IDX 文件路径
    :return: 解析后的数据 (numpy 数组)
    """
    with open(file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_items = int.from_bytes(f.read(4), byteorder='big')

        if magic_number == 2051:  # 图像文件 (magic number: 2051)
            num_rows = int.from_bytes(f.read(4), byteorder='big')
            num_cols = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_items, num_rows, num_cols)
        elif magic_number == 2049:  # 标签文件 (magic number: 2049)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid IDX file format: {file_path}")
    
    return data

def save_images(image_data, label_data, save_dir, prefix):
    """
    将图像数据和标签保存为 PNG 文件
    :param image_data: 图像数据 (numpy 数组, shape: [num_images, height, width])
    :param label_data: 标签数据 (numpy 数组, shape: [num_images])
    :param save_dir: 输出目录
    :param prefix: 文件名前缀 (train/test)
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, (image, label) in enumerate(zip(image_data, label_data)):
        # 图片命名：<prefix>_<index>_<label>.png，例如 train_0_5.png
        img_filename = f"{prefix}_{i}_{label}.png"
        img_path = os.path.join(save_dir, img_filename)

        # 保存图片
        img = Image.fromarray(image)  # 转换为 PIL Image 格式
        img.save(img_path)

        # 打印进度
        if i % 1000 == 0:
            print(f"Processed {i}/{len(image_data)} images...")

def main():
    # 定义路径
    raw_data_dir = os.path.abspath("../../dataset/mnist/raw")
    processed_data_dir = os.path.abspath("../../dataset/mnist/processed")

    # 文件路径
    train_images_path = os.path.join(raw_data_dir, "train-images.idx3-ubyte")
    train_labels_path = os.path.join(raw_data_dir, "train-labels.idx1-ubyte")
    test_images_path = os.path.join(raw_data_dir, "t10k-images.idx3-ubyte")
    test_labels_path = os.path.join(raw_data_dir, "t10k-labels.idx1-ubyte")

    # 输出路径
    train_output_dir = os.path.join(processed_data_dir, "train_data")
    test_output_dir = os.path.join(processed_data_dir, "test_data")

    # 解析训练数据
    print("Reading training data...")
    train_images = read_idx(train_images_path)
    train_labels = read_idx(train_labels_path)

    # 保存训练数据
    print("Saving training images...")
    save_images(train_images, train_labels, train_output_dir, prefix="train")

    # 解析测试数据
    print("Reading testing data...")
    test_images = read_idx(test_images_path)
    test_labels = read_idx(test_labels_path)

    # 保存测试数据
    print("Saving testing images...")
    save_images(test_images, test_labels, test_output_dir, prefix="test")

    print("All images have been processed and saved.")

if __name__ == "__main__":
    main()
