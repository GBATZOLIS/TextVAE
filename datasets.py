# datasets.py

"""
Dataset utilities for loading and preparing image data.
This version is adapted to work with the CLIP image processor's requirements.
"""

import logging
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CelebA

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    A generic PyTorch Dataset for image folders, adapted for the VAE.

    It loads images, applies the necessary transformations to prepare them
    for the model (e.g., resizing and normalizing to [-1, 1]).

    Args:
        root (str | Path): The root directory where the dataset is located.
        image_size (int): The size to which images will be resized.
        split (str): The dataset split to use (e.g., "train", "valid", "test").
                     This is used to select the appropriate subfolder.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        split: Literal["train", "valid", "test"] = "train",
    ):
        super().__init__()
        self.root = Path(root)
        self.image_size = image_size
        self.split = split

        # Using CelebA as the example dataset
        self.dataset = CelebA(
            root=str(self.root), split=self.split, target_type="attr", download=False
        )

        # --- Define Image Transformations ---
        # The model's diffusion VAE expects inputs in the [-1, 1] range.
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.CenterCrop((image_size, image_size)),
                T.ToTensor(),  # Converts to [0, 1] range
                T.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Converts to [-1, 1]
            ]
        )
        logger.info(
            f"Successfully loaded CelebA '{self.split}' split with {len(self.dataset)} samples."
        )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the image at the given index and applies transformations.
        """
        image, _ = self.dataset[idx]
        return self.transform(image)
