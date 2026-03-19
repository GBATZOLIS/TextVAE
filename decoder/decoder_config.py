import os
import torch
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    # --- Model & Local HPC Data Paths ---
    model_id: str = "PixArt-alpha/PixArt-XL-2-512x512"  # Base DiT model
    data_path: str = "/home/rg625/datasets/pixmo_ready.jsonl"
    image_dir: str = "/home/rg625/datasets/pixmo_images"

    # --- Resolution & Sequence Setup ---
    img_size: int = 512  # Diffusers DiTs usually train well at 512 or 256
    max_len: int = 300  # Extended length for T5-XXL

    # --- Checkpoint / Logging ---
    save_dir: str = "/home/rg625/mnt/TextVAE/decoder_checkpoints"
    use_wandb: bool = True

    # --- Training (DiTs need lower learning rates) ---
    batch_size: int = 4  # Keep batch size low due to T5 and VAE memory overhead
    lr: float = 1e-4
    epochs: int = 50
    steps_per_epoch: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- HPC Scaling Optimizations ---
    use_amp: bool = True
    num_workers: int = 8
    prefetch_factor: int = 2

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if not os.path.exists(self.data_path):
            print(f"WARNING: Could not find dataset at {self.data_path}")
