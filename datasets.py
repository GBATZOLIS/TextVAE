# datasets.py
import logging
from pathlib import Path
from typing import Literal
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """A PyTorch Dataset for the CelebA dataset."""

    def __init__(
        self,
        root: str | Path,
        image_size: int,
        split: Literal["train", "valid", "test"],
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.CenterCrop((image_size, image_size)),
                T.ToTensor(),
                # Normalization for SDXL VAE
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        try:
            # Set download=True if you don't have the dataset locally
            self.dataset = CelebA(
                root=str(self.root), split=split, target_type="attr", download=False
            )
            logger.info(
                f"Successfully loaded CelebA '{split}' split with {len(self.dataset)} samples."
            )
        except Exception as e:
            logger.error(
                f"Failed to load or download CelebA. Please check path and connection. Error: {e}"
            )
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Image.Image]:
        image_pil, _ = self.dataset[idx]
        image_tensor = self.transform(image_pil)
        # We return the PIL image as well for the Canny detector
        return image_tensor, image_pil
