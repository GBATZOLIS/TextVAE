import argparse
import os
from torch.utils.data import DataLoader

# Imports matching the generated file structure
from dataset import ImageTextLengthDataset, collate_fn
from encoder.encoder import Encoder
from encoder.encoder_trainer import Trainer
from encoder.encoder_config import EncoderConfig


def train(config: EncoderConfig):
    print("--- Independent Planning Autoencoder Training ---")
    print(f"Device: {config.device} | Batch Size: {config.batch_size}")

    # 1. Dataset
    # We pass the tokenizer model implied by vocab_size (gpt2)
    dataset = ImageTextLengthDataset(
        config.json_path, config.img_dir, image_size=config.img_size
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False,
    )

    # 2. Model
    model = Encoder(
        img_size=config.img_size,
        patch_size=config.patch_size,
        vit_dim=config.vit_dim,
        vit_depth=config.vit_depth,
        vocab_size=config.vocab_size,
        max_len=config.max_len,
    ).to(config.device)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Trainer
    trainer = Trainer(model, loader, None, config)

    # 4. Training Loop
    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config.epochs):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        save_path = os.path.join(config.save_dir, f"model_ep{epoch}.pt")
        trainer.save(save_path)


if __name__ == "__main__":
    # 1. Load Defaults
    config = EncoderConfig()

    # 2. Allow CLI overrides for common paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, help="Override JSON path")
    parser.add_argument("--img_dir", type=str, help="Override Image dir")
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB")
    parser.add_argument("--batch_size", type=int, help="Override batch size")

    args = parser.parse_args()

    # Update config only if args are provided
    if args.json_path:
        config.json_path = args.json_path
    if args.img_dir:
        config.img_dir = args.img_dir
    if args.use_wandb:
        config.use_wandb = True
    if args.batch_size:
        config.batch_size = args.batch_size

    # 3. Run
    train(config)
