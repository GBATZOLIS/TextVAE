from dataclasses import dataclass
import torch


@dataclass
class EncoderConfig:
    # Data & Paths
    img_size: int = 224
    patch_size: int = 16
    vocab_size: int = 50257  # GPT-2
    max_len: int = 64

    # Checkpoint / Logging
    save_dir: str = "./checkpoints"
    # Pointing to your actual data source
    json_path: str = "/home/rg625/mnt/TextVAE/data/captions.json"
    img_dir: str = "/home/rg625/mnt/TextVAE/data/images"
    use_wandb: bool = False

    # Model Params
    vit_dim: int = 512
    vit_depth: int = 6
    heads: int = 8

    # Training
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 250
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
