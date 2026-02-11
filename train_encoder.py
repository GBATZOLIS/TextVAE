import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, Subset

from encoder.encoder_config import EncoderConfig
from dataset import ImageTextLengthDataset, collate_fn
from encoder.encoder import PlanningAutoencoder
from encoder.encoder_trainer import Trainer


# -------------------------
# Utilities
# -------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config_from_args(args) -> EncoderConfig:
    """
    Safely merges CLI overrides into dataclass config.
    Only overrides values explicitly provided.
    """
    config = EncoderConfig()

    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)

    return config


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_path", type=str)
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Build config safely
    config = build_config_from_args(args)

    print("\n===== FINAL CONFIG =====")
    print(config)
    print("========================\n")

    # -------------------------
    # Dataset
    # -------------------------

    if not os.path.exists(config.json_path):
        raise FileNotFoundError(f"JSON file not found: {config.json_path}")

    if not os.path.exists(config.img_dir):
        raise FileNotFoundError(f"Image directory not found: {config.img_dir}")

    full_dataset = ImageTextLengthDataset(config.json_path, config.img_dir)

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty. Aborting.")

    print(f"Dataset size: {len(full_dataset)}")

    # Split
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    split = int(0.9 * len(indices))

    train_ds = Subset(full_dataset, indices[:split])
    val_ds = Subset(full_dataset, indices[split:])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    # -------------------------
    # Model
    # -------------------------

    model = PlanningAutoencoder(config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    # -------------------------
    # Trainer
    # -------------------------

    trainer = Trainer(model, train_loader, val_loader, config)
    start_epoch = 0

    if args.resume is not None:
        trainer.load(args.resume)

    if args.inference_only:
        print("Running inference on validation set...")
        outputs = trainer.inference()

        for i in range(min(10, len(outputs))):
            print(f"\nSample {i}:")
            print(outputs[i])

        return  # Exit without training

    # -------------------------
    # Training Loop
    # -------------------------

    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(start_epoch, config.epochs):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(config.save_dir, f"model_epoch_{epoch+1}.pt")
            trainer.save(save_path, epoch=epoch)


if __name__ == "__main__":
    main()
