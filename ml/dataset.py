import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MentalHealthDataset(Dataset):
    def __init__(self, num_samples=100):
        # Mock data generation
        self.text_data = torch.randn(num_samples, 768)
        self.audio_data = torch.randn(num_samples, 128)
        self.labels = torch.rand(num_samples, 1) # Risk score 0-1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text_data[idx],
            'audio': self.audio_data[idx],
            'label': self.labels[idx]
        }
