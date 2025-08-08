# data/dataset.py

import logging
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """
    A generic PyTorch Dataset for image folders with enhanced logging and error handling.
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
    ):
        super().__init__()

        logger.info(f"Received dataset root path: '{root}'")
        self.root = Path(root).expanduser().resolve()
        logger.info(f"Resolved absolute path to: {self.root}")

        if not self.root.is_dir():
            logger.critical(f"Dataset path is not a valid directory: {self.root}")
            raise FileNotFoundError(f"Directory does not exist: {self.root}")

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        try:
            logger.info("Initializing torchvision.datasets.ImageFolder...")
            self.dataset = ImageFolder(root=str(self.root), transform=self.transform)

            found_classes = self.dataset.classes
            found_class_count = len(found_classes)
            found_image_count = len(self.dataset.imgs)

            logger.info(f"ImageFolder found {found_class_count} class(es).")
            if found_class_count > 0:
                logger.info(f"Classes found: {found_classes[:5]}")

            logger.info(
                f"ImageFolder found a total of {found_image_count} image files."
            )

            if found_image_count == 0:
                logger.error(
                    "The dataset directory appears to be empty or misconfigured."
                )
                logger.error(
                    "Ensure your images are in subdirectories, e.g., "
                    f"'{self.root}/my_class_name/image.jpg'"
                )
                raise FileNotFoundError(f"No images found in {self.root}")

        except Exception as e:
            logger.critical(
                f"A critical error occurred while initializing ImageFolder: {e}",
                exc_info=True,
            )
            raise

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves an item and handles potential corruption by skipping to the next valid item.
        """
        try:
            image, _ = self.dataset[idx]
            return image
        except (UnidentifiedImageError, OSError, IOError) as e:
            bad_file_path = self.dataset.imgs[idx][0]
            logger.warning(
                f"Skipping corrupted or unreadable image file: {bad_file_path}"
            )
            logger.warning(f"  > Error: {e}")
            return self.__getitem__((idx + 1) % len(self))
        except Exception:
            bad_file_path = self.dataset.imgs[idx][0]
            logger.error(
                f"An unexpected error occurred loading image: {bad_file_path}",
                exc_info=True,
            )
            return self.__getitem__((idx + 1) % len(self))
