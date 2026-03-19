import argparse
import os
import random
import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from decoder.decoder_config import DecoderConfig
from decoder_dataset import StreamingDecoderDataset, collate_fn
from decoder.pretrained_decoder import SemanticDecoder
from decoder.decoder_trainer import DecoderTrainer
from decoder.decoder_eval import DecoderEvaluator

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_config_from_args(args) -> DecoderConfig:
    config = DecoderConfig()
    for key, value in vars(args).items():
        if value is not None:
            setattr(config, key, value)
    return config


def main_worker(local_rank, world_size, args):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12356"  # Avoid collision with encoder

    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    set_seed(args.seed + local_rank)

    config = build_config_from_args(args)
    config.device = torch.device(f"cuda:{local_rank}")

    # Initialize Datasets
    val_ds = StreamingDecoderDataset(config)
    train_ds = StreamingDecoderDataset(config)

    # --- THE LIST SLICING FIX ---
    # Reserve the first 500 images exclusively for validation
    val_ds.dataset = val_ds.dataset[:500]

    images_to_skip = 500
    if local_rank == 0:
        print(f"Skipping first {images_to_skip} images for training...")

    # Safely slice the training dataset list
    safe_skip = min(images_to_skip, len(train_ds.dataset) - 1)
    train_ds.dataset = train_ds.dataset[safe_skip:]

    if local_rank == 0:
        print(f"Validation pool: {len(val_ds.dataset)} images")
        print(f"Training pool: {len(train_ds.dataset)} images")

    per_device_batch_size = config.batch_size // world_size

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    model = SemanticDecoder(config).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainer = DecoderTrainer(model, train_loader, val_loader, config)
    evaluator = DecoderEvaluator(model, val_loader, config.device)

    for epoch in range(config.epochs):
        avg_loss = trainer.train_epoch(epoch)

        if local_rank == 0:
            print(f"\n--- Epoch {epoch} Evaluation ---")
            _ = evaluator.compute_metrics(num_batches=2)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            save_path = os.path.join(config.save_dir, f"decoder_epoch_{epoch+1}.pt")
            trainer.save(save_path, epoch=epoch)

        dist.barrier()

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Spawning {world_size} processes for Decoder DDP...")
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
