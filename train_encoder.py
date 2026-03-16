import argparse
import os
import random
import torch
from torch.utils.data import DataLoader

# --- DDP IMPORTS ---
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from encoder.encoder_config import EncoderConfig
from dataset import StreamingDenseCaptionDataset, collate_fn
from encoder.pretrained_encoder import PlanningGPT2
from encoder.encoder_trainer import Trainer
from encoder.encoder_eval import Evaluator

# --- HPC: ENABLING TENSORFLOAT32 (TF32) ---
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

# ==========================================
# Worker Function for each GPU
# ==========================================
def main_worker(local_rank, world_size, args):
    # 1. Manually set environment variables for DDP
    os.environ['MASTER_ADDR'] = '127.0.0.1' # Loopback explicitly prevents resolving issues
    os.environ['MASTER_PORT'] = '12355'

    # 2. Initialize process group
    dist.init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
    
    # Lock this specific process to a specific GPU
    torch.cuda.set_device(local_rank)

    # Offset the seed by the local rank so each GPU augments data differently
    set_seed(args.seed + local_rank)
    
    config = build_config_from_args(args)
    # CRITICAL: Tell your config to use this specific GPU so the Trainer moves tensors correctly
    config.device = torch.device(f"cuda:{local_rank}")

    if local_rank == 0:
        print("\n===== FINAL DDP CONFIG =====")
        print(config)
        print("============================\n")

    start_epoch = 0
    if args.resume is not None:
        if local_rank == 0:
            print(f"Inspecting checkpoint {args.resume} to fast-forward stream...")
        checkpoint = torch.load(args.resume, map_location="cpu")
        start_epoch = checkpoint["epoch"] + 1

    if local_rank == 0: print("Initializing Streaming Datasets...")
    val_ds = StreamingDenseCaptionDataset(config)
    train_ds = StreamingDenseCaptionDataset(config)

    val_ds.dataset = val_ds.dataset.take(500)
    
    images_to_skip = 500 + (start_epoch * config.steps_per_epoch * config.batch_size * world_size)
    
    if local_rank == 0: 
        print(f"Fast-forwarding stream by {images_to_skip:,} images...")
    train_ds.dataset = train_ds.dataset.skip(images_to_skip)

    # --- SHARD DATASET TO PREVENT GPUs FROM DOING DUPLICATE WORK ---
    if hasattr(train_ds.dataset, "shard"):
        train_ds.dataset = train_ds.dataset.shard(num_shards=world_size, index=local_rank)
        if local_rank == 0:
            print(f"Successfully sharded stream across {world_size} GPUs.")

    # --- NEW: CALCULATE PER-GPU BATCH SIZE ---
    assert config.batch_size % world_size == 0, f"Global batch size {config.batch_size} must be divisible by {world_size} GPUs."
    per_device_batch_size = config.batch_size // world_size
    
    if local_rank == 0:
        print(f"Global Batch Size: {config.batch_size} | Per-GPU Batch Size: {per_device_batch_size}")

    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Move model to the specific local GPU
    model = PlanningGPT2(config).to(local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    # Wrap the model in DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainer = Trainer(model, train_loader, val_loader, config)
    
    if args.resume is not None:
        trainer.load(args.resume)

    evaluator = Evaluator(
        model=trainer.model,
        loader=trainer.val_loader,
        tokenizer=trainer.tokenizer,
        device=trainer.device,
    )

    if args.inference_only:
        if local_rank == 0:
            print("Running inference on validation set...")
            trainer.log_predictions(epoch=0, num_samples=10)
        dist.destroy_process_group()
        return

    if local_rank == 0:
        os.makedirs(config.save_dir, exist_ok=True)

    # 3. DISTRIBUTED TRAINING LOOP
    for epoch in range(start_epoch, config.epochs):
        avg_loss = trainer.train_epoch(epoch)
        
        if local_rank == 0:
            print(f"\n--- Epoch {epoch} Evaluation ---")
            _ = trainer.evaluate(evaluator, num_batches=None)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
            save_path = os.path.join(config.save_dir, f"model_epoch_{epoch+1}.pt")
            trainer.save(save_path, epoch=epoch)
            
        dist.barrier()

    dist.destroy_process_group()


# ==========================================
# Main Spawner
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Detect how many GPUs Slurm actually gave us
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print(f"WARNING: Found {world_size} GPU(s). DDP is meant for 2+ GPUs.")
        print("If you meant to use 1 GPU, just run this normally without DDP wrappers.")

    print(f"Spawning {world_size} processes for DDP...")
    
    # This launches main_worker() on 'world_size' number of GPUs
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))

if __name__ == "__main__":
    main()