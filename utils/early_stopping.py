import torch


class EarlyStopping:
    """
    Early stops the training if the monitored metric doesn't improve after a given patience.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0, save_path="best_model.pth", monitor='val_loss'):
        """
        Args:
            patience (int): 当验证集指标在这段 patience 内没有提升，就会触发早停。
            verbose (bool): 如果为 True，会打印提示信息。
            delta (float): 判断“显著改善”的阈值。
            save_path (str): 验证集指标最优时，保存模型的路径。
            monitor (str): 监控指标，可选 'val_loss' 或 'val_acc'。
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.monitor = monitor

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric_value = None

    def __call__(self, metric_value, model):
        # 根据 monitor 不同，score 处理方式不同
        if self.monitor == 'val_loss':
            # 越低越好，用负数让“更优”时分数更大
            score = -metric_value
        else:
            # 比如监控准确率，越高越好，直接用正值
            score = metric_value

        # 用于打印旧值 -> 新值
        old_best = self.best_metric_value

        # 第一次调用时初始化
        if self.best_score is None:
            self.best_score = score
            self.best_metric_value = metric_value
            self.save_checkpoint(metric_value, model, old_best)  # 传入 old_best
        # 若没有达到显著改善
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        # 若有显著改善
        else:
            self.best_score = score
            self.best_metric_value = metric_value
            self.save_checkpoint(metric_value, model, old_best)
            self.counter = 0

    def save_checkpoint(self, metric_value, model, old_best):
        """当监控指标变得更好时，保存模型，并打印提示信息。"""
        if self.verbose:
            if self.monitor == 'val_loss':
                # 第一次没有 old_best 时，用 inf 表示
                old_val_str = f"{old_best:.6f}" if old_best is not None else "inf"
                print(f"Validation loss decreased ({old_val_str} --> {metric_value:.6f}). Saving model ...")
            else:
                # 第一次没有 old_best 时，用 0.000000 表示
                old_val_str = f"{old_best:.6f}" if old_best is not None else "0.000000"
                print(f"Validation accuracy increased ({old_val_str} --> {metric_value:.6f}). Saving model ...")

        torch.save(model.state_dict(), self.save_path)
