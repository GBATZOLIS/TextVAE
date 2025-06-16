"""Dataset utilities. Replace the dummy dataset with your actual data‑loading pipeline."""
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class DummyImageDataset(Dataset):
    """Synthetic dataset that returns random images in [-1, 1]."""

    def __init__(self, length: int = 1000, image_size: int = 256):
        self.length = length
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randn(3, 256, 256)
