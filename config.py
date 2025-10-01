# config.py

"""Global configuration dataclass used across the project."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class VAEConfig:
    # ---------- image ----------
    image_size: int = 256
    image_channels: int = 3
    patch_size: int = 16

    # ---------- Vision Encoder Selection ----------
    # Type of vision encoder to use. Options: "vit", "swin"
    vision_encoder_type: str = "swin"

    # ---------- ViT Encoder (if type is 'vit') ----------
    encoder_dim: int = 768
    encoder_depth: int = 12
    encoder_heads: int = 12
    encoder_mlp_dim: int = 3072
    encoder_dropout: float = 0.1

    # ---------- Swin Transformer V2 Configs (if type is 'swin') ----------
    # Note: embed_dim is shared with vit's encoder_dim
    swin_depths: List[int] = field(default_factory=lambda: [2, 2, 6, 2])
    swin_num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    swin_window_size: int = 8
    swin_mlp_ratio: float = 4.0
    swin_drop_path_rate: float = 0.1

    # ---------- text decoder ----------
    text_decoder_dim: int = 768  # Should match encoder_dim
    text_decoder_depth: int = 6
    text_decoder_heads: int = 12
    text_decoder_mlp_dim: int = 3072
    text_decoder_dropout: float = 0.1
    max_text_length: int = 77
    vocab_size: int = 50257  # GPT-2 vocab size

    # ---------- gumbel softmax ----------
    gumbel_temperature: float = 1.0
    gumbel_temperature_min: float = 0.1
    gumbel_anneal_rate: float = 3e-5
    use_straight_through: bool = True

    # ---------- diffusion ----------
    diffusion_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # ---------- optimisation ----------
    kl_weight: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    amp: bool = True  # Automatic mixed precision

    # ---------- LLM prior ----------
    llm_model_name: str = "gpt2"
    freeze_llm: bool = True

    # ---------- misc ----------
    seed: int = 42
    log_every: int = 100
    save_every_epochs: int = 5

    # Dynamically tracked fields (do not set manually)
    global_step: int = field(default=0, init=False)
