from dataclasses import dataclass
import torch


@dataclass
class EncoderConfig:
    # Data & Paths
    hf_dataset_path: str = "allenai/PixMo-Cap"  
    hf_cache_dir: str = "/rds/user/rg625/hpc-work/hf_cache" # <-- ADD THIS LINE
    img_size: int = 224
    vocab_size: int = 50257
    max_len: int = 256

    # Checkpoint / Logging
    save_dir: str = "./checkpoints"
    use_wandb: bool = True

    # Training
    batch_size: int = 192
    lr: float = 5e-4
    epochs: int = 100
    steps_per_epoch: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- HPC Scaling Optimizations ---
    use_amp: bool = True  # Automatic Mixed Precision (BFloat16/FP16)
    num_workers: int = 4  # Parallel CPU workers for downloading data
    prefetch_factor: int = 2  # Pre-load batches into RAM ahead of the GPU
