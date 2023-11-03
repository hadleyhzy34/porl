import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pdb

class CustomDataset(Dataset):
    def __init__(self, device=torch.device('cpu')):
        self.device = device

        list = os.listdir('checkpoint/')
        self.length = len(list) * 100

    def __getitem__(self, index):
        # pdb.set_trace()
        file_index = index // 100
        row_index = index % 100
        data = np.loadtxt(f"checkpoint/dataset_{file_index}.csv", delimiter=',',dtype=float)
        # data = torch.load(f'checkpoint/dataset_{file_index}.pt')[row_index]
        if data.max() < 0.1 or data.shape[0] != 72800:
            pdb.set_trace()
        data = data.reshape(100,-1)[row_index]

        # data = np.concatenate([
        #     data['state'],
        #     [data['reward']],
        #     data['next_state'],
        #     [data['done']],
        #     data['action']
        #     ])
        data = torch.from_numpy(data.astype(np.float32)).to(self.device)

        return data

    def __len__(self):
        return self.length
