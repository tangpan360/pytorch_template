"""Utils package containing various helper functions and classes for data processing, training, and evaluation."""

# 导入包中的各个模块，并将它们的主要类或函数赋值给包级别的名称空间
# 这样用户可以直接从 utils 包中导入这些组件，而不需要知道具体的模块名
from .cifar10_dataset import Cifar10Dataset
from .early_stopping import EarlyStopping
from .seed_utils import set_seed
from .time_utils import format_time
from .trainer import Trainer

# 定义 __all__ 列表，明确指出哪些名称是本包的公共 API
# 这有助于控制通过 from utils import * 导入的内容，并文档化包的主要接口
__all__ = [
    'Cifar10Dataset',
    'EarlyStopping',
    'set_seed',
    'format_time',
    'Trainer'
]

# 定义包的版本号，方便其他地方引用
__version__ = "0.1.0"
