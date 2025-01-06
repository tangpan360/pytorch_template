#!/bin/bash

# 获取脚本所在目录的绝对路径 (更加兼容的方式)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# 设置项目根目录为 PYTHONPATH 环境变量的一部分
export PYTHONPATH="${SCRIPT_DIR}":$PYTHONPATH

# 创建 logs 目录（如果不存在）
mkdir -p "${SCRIPT_DIR}/logs"

# 使用 script 命令记录 Python 脚本的执行过程到日志文件中。
# -q 选项使 script 命令安静运行，不会显示会话启动信息。
# -c 'command' 指定要执行的命令，在这里我们执行 python main.py。
script -q -c 'python main.py' ./logs/train_log.txt
