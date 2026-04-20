import torch
from torch.utils.data import Dataset


class NSForcingDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path, weights_only=False)
        self.x = data['x'].float()  # (N, 128, 128)
        self.y = data['y'].float()  # (N, 128, 128)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
