import torch
import pandas as pd

from torch.utils.data import Dataset
from transformers import BertTokenizer


class AGNewsDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化 AGNewsDataset。

        Args:
            data_path (str): 数据集路径 (parquent 文件)。
            tokenizer (transformers.PreTrainedBertTokenizer): BERT 分词器。
            max_length (int): 最大文本长度。
        """
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]  # 文本列名为 "text"
        label = self.data.iloc[idx]["label"]  # 标签列名为 "label"

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
