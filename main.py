# main.py

"""
Entry-point script to configure and start training the Text-Conditioned VAE.

This script handles:
1. Parsing command-line arguments.
2. Setting up the configuration object.
3. Initializing datasets and dataloaders.
4. Creating and running the trainer.
"""
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from config import VAEConfig
from datasets import ImageDataset
from trainer import TextVAETrainer


def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Train a Text-Conditioned VAE Proof-of-Concept."
    )

    # --- Training Arguments ---
    p.add_argument("--epochs", type=int, help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, help="Batch size per device.")
    p.add_argument("--learning_rate", type=float, help="Peak learning rate.")
    p.add_argument("--kl_weight", type=float, help="Weight for the KL divergence loss.")
    p.add_argument(
        "--disable_amp", action="store_true", help="Disable Automatic Mixed Precision."
    )
    p.add_argument(
        "--work_dir",
        type=str,
        default="./runs",
        help="Directory for checkpoints and logs.",
    )
    p.add_argument(
        "--dataset_path", type=str, help="Path to the root of the image dataset."
    )

    return p.parse_args()


def main():
    """
    Main function to set up and run the training process.
    """
    # 1. Configure logging and parse arguments.
    setup_logging()
    args = parse_args()

    # 2. Create the configuration object, overriding defaults with CLI args.
    # This allows for flexible experimentation without changing the config file.
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    if "disable_amp" in overrides:
        overrides["use_amp"] = not overrides.pop("disable_amp")

    cfg = VAEConfig(**overrides)
    torch.manual_seed(cfg.seed)

    # 3. Create datasets and dataloaders.
    logging.info(f"Setting up dataset from {cfg.dataset_path}...")
    try:
        train_ds = ImageDataset(
            root=cfg.dataset_path, image_size=cfg.image_size, split="train"
        )
        val_ds = ImageDataset(
            root=cfg.dataset_path, image_size=cfg.image_size, split="valid"
        )
    except Exception:
        logging.error(
            f"Failed to load dataset at path: {cfg.dataset_path}", exc_info=True
        )
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, num_workers=4, pin_memory=True
    )
    logging.info(
        f"Training with {len(train_ds)} samples, validating with {len(val_ds)} samples."
    )

    # 4. Initialize and run the trainer.
    try:
        trainer = TextVAETrainer(cfg, work_dir=args.work_dir)
        trainer.fit(train_loader, val_loader)
    except Exception:
        logging.error("An unrecoverable error occurred during training.", exc_info=True)


if __name__ == "__main__":
    main()
