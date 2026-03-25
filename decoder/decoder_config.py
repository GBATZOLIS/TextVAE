import os
import torch
from dataclasses import dataclass


@dataclass
class DecoderConfig:
    # --- Model & Local HPC Data Paths ---
    # model_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-512x512"
    model_id: str = "PixArt-alpha/PixArt-XL-2-512x512"
    data_path: str = "/home/rg625/datasets/pixmo_ready.jsonl"
    image_dir: str = "/home/rg625/datasets/pixmo_images"

    # --- Resolution & Sequence Setup ---
    img_size: int = 512
    max_len: int = 384  # Enforced massive sequence length!

    # --- Checkpoint / Logging ---
    save_dir: str = "/home/rg625/mnt/TextVAE/decoder_checkpoints"
    use_wandb: bool = True

    # --- Training ---
    batch_size: int = 1
    lr: float = 2e-4
    epochs: int = 100
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
