import os
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)  # transform就是数据增强，对每一个数据都执行
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.num_items = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):  # 检查self.raw_dir目录下是否存在raw_file_names()属性方法返回的每个文件
        # 如有文件不存在，则调用download()方法执行原始文件下载
        return ['yoochoose-clicks.dat', 'yoochoose-buys.dat']

    @property
    def processed_file_names(self):  # 检查self.processed_dir目录下是否存在self.processed_file_names属性方法返回的所有文件，没有就会走process
        return ['yoochoose_click_binary_1M_sess.dataset', 'num_items.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        # process by session_id
        click_df = pd.read_csv(os.path.join(self.raw_dir, 'yoochoose-clicks.dat'))
        click_df.columns = ['session_id', 'timestamp', 'item_id', 'category']

        sampled_session_id = np.random.choice(click_df['session_id'].unique(), size=10000, replace=False)
        click_df = click_df.loc[click_df['session_id'].isin(sampled_session_id)]

        buy_df = pd.read_csv(os.path.join(self.raw_dir, 'yoochoose-buys.dat'))
        buy_df.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

        item_encoder = LabelEncoder()
        click_df['item_id'] = item_encoder.fit_transform(click_df['item_id'])

        max_item_id = click_df['item_id'].max()
        torch.save(max_item_id + 1, self.processed_paths[1])

        grouped = click_df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features

            # y = torch.FloatTensor([group.label.values[0]])
            y = torch.FloatTensor([1.0 if session_id in buy_df['session_id'].values else 0.0])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])