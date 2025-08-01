# datasets.py

"""
Dataset utilities for loading image data.
This implementation provides a PyTorch Dataset for the CelebA dataset.

CelebA Dataset Structure:
To use this script, you should download the CelebA dataset and place it in a root
directory. The expected structure is:

<root_dir>/
├── celeba/
│   ├── img_align_celeba/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── list_attr_celeba.txt
│   └── list_eval_partition.txt
└── ...

You can download the dataset from its official website or use torchvision's
built-in downloader.
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


class CelebADataset(Dataset):
    """
    PyTorch Dataset for the CelebA (CelebFaces Attributes) dataset.

    This class handles loading images and applying transformations. It uses the
    official train/validation/test splits defined in `list_eval_partition.txt`.

    Args:
        root (str | Path): The root directory where the CelebA dataset is located
                           or where it should be downloaded.
        split (str): The dataset split to use, one of "train", "valid", or "test".
        image_size (int): The size to which images will be resized (image_size x image_size).
        download (bool): If True, downloads the dataset from the internet if it's not
                         available at the root directory.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "valid", "test"] = "train",
        image_size: int = 256,
        download: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        # --- Define Image Transformations ---
        # 1. Resize the image to a square of `image_size`.
        # 2. Convert the PIL Image to a PyTorch tensor (scales to [0, 1]).
        # 3. Normalize the tensor to the range [-1, 1], which is a common
        #    practice for VAEs and GANs.
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # --- Load the CelebA dataset using torchvision's utility ---
        # This conveniently handles downloading and parsing the split information.
        try:
            self.celeba = CelebA(
                root=str(self.root),
                split=self.split,
                target_type="attr",  # We don't use attributes here, but it's required
                download=download,
            )
            logger.info(
                f"Successfully loaded CelebA '{self.split}' split with {len(self.celeba)} samples."
            )
        except (RuntimeError, FileNotFoundError) as e:
            logger.error(
                f"Could not load or download CelebA dataset at {self.root}. "
                "Please check the path or enable download=True."
            )
            raise e

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset split."""
        return len(self.celeba)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves the image at the given index, applies transformations, and returns it.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            torch.Tensor: The transformed image as a tensor.
        """
        # `self.celeba[idx]` returns a tuple (image, attributes). We only need the image.
        image, _ = self.celeba[idx]

        # Apply the defined transformations
        return self.transform(image)


# --- Example Usage (for testing the dataset) ---
if __name__ == "__main__":
    # This block will only run when the script is executed directly
    # It serves as a simple test to ensure the dataset is working correctly.

    # Configure basic logging for the test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    DATA_DIR = "./data"  # Directory to store the dataset
    IMAGE_SIZE = 128

    logger.info("--- Testing CelebADataset ---")
    try:
        # 1. Instantiate the dataset for the training split
        train_dataset = CelebADataset(
            root=DATA_DIR, split="train", image_size=IMAGE_SIZE, download=True
        )

        # 2. Get a single sample from the dataset
        sample_image = train_dataset[0]

        # 3. Log information about the sample
        logger.info(f"\nDataset length (train split): {len(train_dataset)}")
        logger.info(f"Sample image shape: {sample_image.shape}")
        logger.info(f"Sample image dtype: {sample_image.dtype}")
        logger.info(f"Sample image min value: {sample_image.min():.2f}")
        logger.info(f"Sample image max value: {sample_image.max():.2f}")

        # Verify the shape is as expected: [Channels, Height, Width]
        assert sample_image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        # Verify the normalization to [-1, 1] is working
        assert sample_image.min() >= -1.0 and sample_image.max() <= 1.0

        logger.info("\n✅ Test passed successfully!")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
