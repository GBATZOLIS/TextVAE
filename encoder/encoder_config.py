from dataclasses import dataclass
import torch


@dataclass
class EncoderConfig:
    # --- Data Paths ---
    json_path: str = "data/captions.json"
    img_dir: str = "data/images/"
    save_dir: str = "./checkpoints"

    # --- Model Architecture ---
    vit_dim: int = 512
    vit_depth: int = 8
    patch_size: int = 16
    img_size: int = 224
    vocab_size: int = 50257  # GPT-2 default
    max_len: int = 100

    # --- Training Hyperparameters ---
    batch_size: int = 1
    epochs: int = 10
    lr: float = 1e-4
    dropout: float = 0.1

    # --- System ---
    use_wandb: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
