def format_time(seconds: float) -> str:
    """
    将秒数转换为 HH:MM:SS 的格式，便于可读输出。

    Args:
        seconds (float): 总秒数。

    Returns:
        str: 格式化后的时间字符串。
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == '__main__':
    # 测试数据点
    test_seconds = [0, 61, 3661, 7200, 86400]

    # 测试 format_time 函数
    for sec in test_seconds:
        print(f"{sec} seconds is formatted as {format_time(sec)}")
