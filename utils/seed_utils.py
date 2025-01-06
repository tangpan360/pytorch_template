import random
import numpy as np
import os
import torch


def set_seed(seed=42):
    # 下面两个常规设置了，用来np和random的话要设置
    random.seed(seed)


    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU训练需要设置这个

    # torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。


if __name__ == '__main__':
    # 设定随机种子
    set_seed(42)  # 设定随机种子
