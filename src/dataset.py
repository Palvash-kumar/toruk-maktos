import torch
from torch.utils.data import Dataset

class LandslideDataset(Dataset):

    def __init__(
        self,
        features,
        labels
    ):

        self.X = torch.tensor(
            features.reshape(-1, features.shape[-1]),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            labels.reshape(-1),
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.y[idx]
        )