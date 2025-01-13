import os
import requests
import tarfile
from urllib.parse import urlparse

# 定义下载链接和目标文件夹
url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
target_folder = "../../dataset/cifar10/raw"

# 如果目标文件夹不存在，则创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 解析URL获取文件名
parsed_url = urlparse(url)
filename = os.path.basename(parsed_url.path)

# 定义本地文件路径
local_file_path = os.path.join(target_folder, filename)

# 下载文件
print(f"开始下载 {filename} 到 {local_file_path}")
response = requests.get(url, stream=True)
with open(local_file_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:  # 过滤掉保持连接的空chunk
            f.write(chunk)
print("下载完成")

# 解压文件
print("开始解压文件...")
with tarfile.open(local_file_path, "r:gz") as tar:
    tar.extractall(path=target_folder)
print("解压完成")

# 可选：删除原始压缩文件以节省空间
# os.remove(local_file_path)
# print("原始压缩文件已删除")