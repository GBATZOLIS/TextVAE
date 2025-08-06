# main.py
import logging
import torch
import os
from torch.utils.data import DataLoader
from pathlib import Path

from config import config, AppConfig
from trainer import TextVAETrainer
from datasets import ImageDataset


def setup_logging():
    """Configures the root logger for clean and informative output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_dataloaders(cfg: AppConfig) -> tuple[DataLoader, DataLoader]:
    """Initializes and returns the training and validation dataloaders."""
    expanded_path = os.path.expanduser(cfg.data.dataset_path)

    if not os.path.exists(expanded_path):
        raise FileNotFoundError(
            f"Dataset root path not found. Please ensure '{expanded_path}' exists."
        )

    train_dataset = ImageDataset(
        root=expanded_path, image_size=cfg.data.image_resolution, split="train"
    )
    val_dataset = ImageDataset(
        root=expanded_path, image_size=cfg.data.image_resolution, split="valid"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.training.batch_size, num_workers=4
    )

    logging.info(
        f"Training with {len(train_dataset)} images, validating with {len(val_dataset)} images."
    )
    return train_loader, val_loader


def main():
    """Main function to configure and run the training pipeline."""
    setup_logging()

    # --- System Setup ---
    torch.manual_seed(config.system.seed)
    output_dir = Path(config.system.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output will be saved to: {output_dir}")

    # --- Data Loading ---
    try:
        train_loader, val_loader = create_dataloaders(config)
    except Exception as e:
        logging.error(f"Failed to create dataloaders: {e}", exc_info=True)
        return

    # --- Training ---
    try:
        trainer = TextVAETrainer(config)
        trainer.fit(train_loader, val_loader)
    except Exception as e:
        logging.error(
            f"An unrecoverable error occurred during training: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
