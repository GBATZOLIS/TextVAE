import os
import torch
from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    # Data & Paths
    hf_dataset_path: str = "allenai/PixMo-Cap"
    # Explicitly define the absolute path or expand the user directory safely
    hf_cache_dir: str = field(
        default_factory=lambda: os.path.expanduser("/home/rg625/datasets/hf_cache")
    )

    # SOTA Fix: Bumped resolution for dense captioning grounding
    # DINOv2 patch size is 14. 336 / 14 = 24. (24x24 = 576 patches)
    img_size: int = 224
    vocab_size: int = 50257
    max_len: int = 256

    # Checkpoint / Logging
    save_dir: str = "./encoder_checkpoints"
    use_wandb: bool = True

    # Training
    batch_size: int = 16
    lr: float = 5e-4
    epochs: int = 100
    steps_per_epoch: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- HPC Scaling Optimizations ---
    use_amp: bool = True

    # SOTA Fix: Massive worker bump to prevent HTTP blocking from starving the GPU
    num_workers: int = 16

    # Bump prefetch so the RAM buffer absorbs network latency spikes
    prefetch_factor: int = 4

    def __post_init__(self):
        # Safety check to ensure paths exist
        os.makedirs(self.hf_cache_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
