# main.py

"""Entry-point script to start training the TextVAE model."""
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# from datasets import CelebADataset  <-- REMOVE THIS
from datasets import MPI3DDataset  # <-- ADD THIS
from trainer import TextVAETrainer
from config import VAEConfig


# ... (setup_logging and parse_args functions remain the same) ...
def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Train a Text-Conditioned VAE.")
    p.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size per device.")
    p.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Peak learning rate."
    )
    p.add_argument(
        "--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)."
    )
    p.add_argument(
        "--work_dir",
        type=str,
        default="./runs",
        help="Directory to save checkpoints and logs.",
    )
    p.add_argument(
        "--vision_encoder_type",
        type=str,
        default="swin",
        choices=["vit", "swin"],
        help="Type of vision encoder.",
    )
    return p.parse_args()


def main():
    """
    Main function to set up and run the training process.
    """
    torch.autograd.set_detect_anomaly(True)
    setup_logging()
    args = parse_args()
    cfg = VAEConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        amp=args.amp,
        vision_encoder_type=args.vision_encoder_type,
    )
    logging.info(f"Configuration loaded: {cfg}")
    # The root directory for your datasets
    data_root = Path("./datasets").expanduser()

    # 3. Create datasets and dataloaders.
    logging.info(f"Setting up MPI3D dataset from {data_root}...")
    # **MODIFIED**: Use MPI3DDataset
    train_ds = MPI3DDataset(root=data_root, image_size=cfg.image_size, split="train")
    # For MPI3D, we can use the same data for validation or create a manual split later
    val_ds = MPI3DDataset(root=data_root, image_size=cfg.image_size, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
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
        trainer.fit(train_loader, val_loader, epochs=args.epochs)
    except Exception:
        logging.error("An error occurred during training.", exc_info=True)


if __name__ == "__main__":
    main()
