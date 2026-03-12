import argparse
import os
import random
import torch
from torch.utils.data import DataLoader

from encoder.encoder_config import EncoderConfig
from dataset import StreamingDenseCaptionDataset, collate_fn
from encoder.pretrained_encoder import PlanningGPT2
from encoder.encoder_trainer import Trainer
from encoder.encoder_eval import Evaluator

# --- HPC: ENABLING TENSORFLOAT32 (TF32) ---
# Cambridge HPC A100s will process matmuls ~3x faster with this flag enabled
torch.set_float32_matmul_precision("high")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config_from_args(args) -> EncoderConfig:
    config = EncoderConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    config = build_config_from_args(args)

    print("\n===== FINAL CONFIG =====")
    print(config)
    print("========================\n")

    start_epoch = 0
    if args.resume is not None:
        print(f"Inspecting checkpoint {args.resume} to fast-forward stream...")
        checkpoint = torch.load(args.resume, map_location="cpu")
        start_epoch = checkpoint["epoch"] + 1

    print("Initializing Streaming Datasets...")
    val_ds = StreamingDenseCaptionDataset(config)
    train_ds = StreamingDenseCaptionDataset(config)

    val_ds.dataset = val_ds.dataset.take(500)
    images_to_skip = 500 + (start_epoch * config.steps_per_epoch * config.batch_size)
    print(f"Fast-forwarding stream by {images_to_skip:,} images...")
    train_ds.dataset = train_ds.dataset.skip(images_to_skip)

    # --- HPC: ADVANCED DATALOADER CONFIG ---
    # pin_memory allows direct async transfer to GPU via PCIe
    # prefetch_factor keeps batches in RAM ahead of time, hiding network latency
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = PlanningGPT2(config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (LoRA + Mapper): {trainable_params:,}")

    trainer = Trainer(model, train_loader, val_loader, config)
    if args.resume is not None:
        trainer.load(args.resume)

    evaluator = Evaluator(
        model=trainer.model,
        loader=trainer.val_loader,
        tokenizer=trainer.tokenizer,
        device=trainer.config.device,
    )

    if args.inference_only:
        print("Running inference on validation set...")
        trainer.log_predictions(epoch=0, num_samples=10)
        return

    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(start_epoch, config.epochs):
        avg_loss = trainer.train_epoch(epoch)
        print(f"\n--- Epoch {epoch} Evaluation ---")
        _ = trainer.evaluate(evaluator, num_batches=None)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        save_path = os.path.join(config.save_dir, f"model_epoch_{epoch+1}.pt")
        trainer.save(save_path, epoch=epoch)


if __name__ == "__main__":
    main()
