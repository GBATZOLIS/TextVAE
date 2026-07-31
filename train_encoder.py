import os
import argparse
import random
import tempfile

import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from encoder.encoder_config import EncoderConfig
from dataset import ProgressiveDataset
from encoder.pretrained_encoder import ProgressiveSemanticEncoder
from encoder.encoder_trainer import Trainer

# Optional: Set sharing strategy to prevent C-level dataloader aborts in multi-processing
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
# --- FIX FOR HPC TMP DIR ERROR ---
# Create a local temporary directory in your project folder
local_tmp = os.path.abspath("./local_tmp")
os.makedirs(local_tmp, exist_ok=True)
os.environ["TMPDIR"] = local_tmp
tempfile.tempdir = local_tmp
# ---------------------------------


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main_worker(rank: int, world_size: int, args: argparse.Namespace):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Offset seed by rank to prevent identical augmentations across GPUs
    set_seed(42 + rank)

    config = EncoderConfig()

    # Apply command line arguments to config (only override if explicitly True)
    if hasattr(args, "wandb") and args.wandb:
        config.use_wandb = True
    if hasattr(args, "eval_epochs") and args.eval_epochs is not None:
        config.eval_every_n_epochs = args.eval_epochs
    if hasattr(args, "eval_steps") and args.eval_steps is not None:
        config.eval_every_m_steps = args.eval_steps

    if rank == 0:
        print("Initializing Tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ProgressiveDataset(config, tokenizer)

    assert (
        config.batch_size % world_size == 0
    ), "Global batch size must be divisible by world size!"
    local_batch_size = config.batch_size // world_size

    # IterableDatasets handle their own sharding, so NO DistributedSampler is used.
    dataloader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    if rank == 0:
        print("Initializing Progressive Semantic Encoder...")
    model = ProgressiveSemanticEncoder(config, tokenizer).to(rank)

    # find_unused_parameters=True is often required when using LoRA + custom components
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Passing dataloader as val_loader (in production, instantiate a separate Validation Dataset)
    trainer = Trainer(model, dataloader, config, rank, val_loader=dataloader)

    if rank == 0 and config.use_wandb:
        import wandb

        wandb.init(project="planning-autoencoder", config=vars(config))

    for epoch in range(config.epochs):
        avg_loss = trainer.train_epoch(epoch)

        # End of Epoch Evaluation Check
        if config.eval_every_n_epochs > 0 and epoch % config.eval_every_n_epochs == 0:
            if rank == 0:
                print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
                trainer.log_samples(epoch)  # Trigger Evaluation
                trainer.save_checkpoint(epoch)

        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Progressive Semantic Encoder")
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights and Biases logging"
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=1,
        help="Evaluate every N epochs (0 to disable)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every M steps (0 to disable)",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size > 0:
        print(f"Spawning {world_size} distributed processes...")
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    else:
        print("ERROR: No GPUs found. This cluster pipeline requires CUDA.")
