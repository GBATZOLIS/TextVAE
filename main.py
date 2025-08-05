# main.py

"""
Entry-point script to configure and start training the End-to-End Swin-VAE.
"""
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from config import VAEConfig
from trainer import TextVAETrainer
from datasets import ImageDataset


def setup_logging(work_dir: Path):
    """Configures logging to file and console."""
    log_file = work_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="Train an End-to-End Text-Conditioned VAE.")
    p.add_argument("--epochs", type=int, help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, help="Batch size per device.")
    p.add_argument("--learning_rate", type=float, help="Peak learning rate.")
    p.add_argument("--kl_weight", type=float, help="Weight for the KL divergence loss.")
    p.add_argument(
        "--disable_amp", action="store_true", help="Disable Automatic Mixed Precision."
    )
    p.add_argument("--work_dir", type=str, help="Directory for checkpoints and logs.")
    # p.add_argument("--dataset_path", type=str, required=True, help="Path to the root of the image dataset (e.g., '~/datasets/my_images').")
    p.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    return p.parse_args()


def main():
    """Main function to set up and run the training process."""
    args = parse_args()

    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["resume_from"]
    }
    if "disable_amp" in overrides:
        overrides["use_amp"] = not overrides.pop("disable_amp")

    cfg = VAEConfig(**overrides)
    torch.manual_seed(cfg.seed)

    work_dir = Path(cfg.work_dir).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(work_dir)

    # --- Create datasets and dataloaders ---
    dataset_path_str = cfg.dataset_path
    logging.info(f"Attempting to load dataset from provided path: {dataset_path_str}")

    # FIX: Explicitly expand the user's home directory, resolve to an absolute path, and add robust error handling.
    try:
        # First, check if the provided path actually exists.
        dataset_path = Path(dataset_path_str).expanduser().resolve(strict=True)
        logging.info(f"Setting up dataset from resolved absolute path: {dataset_path}")
    except FileNotFoundError:
        # This error is caught if the path literally does not exist.
        logging.error(
            f"The provided dataset path does not exist: {Path(dataset_path_str).expanduser()}"
        )
        return

    try:
        # Now, try to load the dataset using ImageFolder
        full_dataset = ImageDataset(root=dataset_path)

        if len(full_dataset) == 0:
            # This error means the path exists, but ImageFolder found no images.
            # This is the most likely user error.
            logging.error(f"No images found in the dataset directory: {dataset_path}")
            logging.error(
                "PyTorch's ImageFolder expects images to be in subdirectories (e.g., root/class/image.jpg)."
            )
            logging.error(
                f"Based on your file listing, you might need to provide a more specific path. For CelebA, try: --dataset_path '{dataset_path / 'celeba' / 'img_align_celeba'}' or '{dataset_path / 'img_align_celeba'}'"
            )
            return

        # Create a 90/10 train/validation split
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

    except Exception as e:
        # This catches other potential errors during dataset loading.
        logging.error(
            f"Failed to load dataset from '{dataset_path}'. Error: {e}", exc_info=True
        )
        logging.error(
            "This could be due to corrupted files or an unexpected folder structure."
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

    # --- Initialize and run the trainer ---
    try:
        trainer = TextVAETrainer(cfg, work_dir=str(work_dir))
        if args.resume_from:
            resume_path = Path(args.resume_from).expanduser()
            trainer.load_checkpoint(str(resume_path))

        trainer.fit(train_loader, val_loader)
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user. Exiting.")
    except Exception as e:
        logging.error(
            f"An unrecoverable error occurred during training: {e}", exc_info=True
        )


if __name__ == "__main__":
    main()
