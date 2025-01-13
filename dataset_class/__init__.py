"""Utils package containing various dataset classes for dataset processing."""

# 导入包中的各个模块，并将它们的主要类或函数赋值给包级别的名称空间
# 这样用户可以直接从 dataset_class 包中导入这些组件，而不需要知道具体的模块名
from .cifar10_dataset import Cifar10Dataset
from .mnist_dataset import MnistDataset

# 定义 __all__ 列表，明确指出哪些名称是本包的公共 API
# 这有助于控制通过 from dataset_class import * 导入的内容，并文档化包的主要接口
__all__ = [
    'Cifar10Dataset',
    'MnistDataset'
]

# 定义包的版本号，方便其他地方引用
__version__ = "0.1.0"
