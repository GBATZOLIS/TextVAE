# main.py
import logging
import torch
import os
from torch.utils.data import DataLoader
from pathlib import Path

from config import config
from trainer import CodecTrainer
from datasets import ImageDataset  # Assuming you have a datasets.py


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
    )


def main():
    setup_logging()
    torch.manual_seed(config.system.seed)
    Path(config.system.output_dir).mkdir(parents=True, exist_ok=True)

    # --- Data Loading ---
    expanded_path = os.path.expanduser(config.data.dataset_path)
    train_dataset = ImageDataset(
        root=expanded_path, image_size=config.data.image_resolution, split="train"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True
    )

    # --- Training ---
    trainer = CodecTrainer(config)
    trainer.fit(train_loader)


if __name__ == "__main__":
    main()
