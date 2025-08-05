# config.py

"""
Centralized configuration for the Text-Conditioned VAE model.

Using a dataclass provides type hints, default values, and a single, clear
source of truth for all hyperparameters, making experiments easier to manage
and reproduce.
"""

import logging
from dataclasses import dataclass

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VAEConfig:
    """Configuration for the Text-Conditioned VAE."""

    # --- Model Architecture ---
    # Defines the pre-trained models to use. These are chosen for their SOTA performance.
    vision_encoder_model: str = "openai/clip-vit-large-patch14"
    llm_prior_model: str = "gpt2"
    # Using SD 1.5 as the decoder provides a powerful, pre-trained generator.
    diffusion_decoder_model: str = "runwayml/stable-diffusion-v1-5"

    # --- Text Decoder Specifics ---
    # These parameters control the architecture of the trainable text decoder.
    text_decoder_dim: int = (
        768  # Must match the dimension of the LLM's embeddings (GPT-2 base is 768)
    )
    text_decoder_heads: int = 12  # Number of attention heads
    text_decoder_depth: int = 12  # Number of transformer blocks
    text_decoder_mlp_dim: int = 3072  # Dimension of the feed-forward layer
    max_text_length: int = (
        77  # Max length of the generated text sequence (CLIP's default)
    )

    # --- Training Hyperparameters ---
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    kl_weight: float = 1e-4  # Weight for the KL divergence term in the VAE loss
    seed: int = 42
    use_amp: bool = True  # Use Automatic Mixed Precision for faster training
    work_dir: str = "./runs"
    save_every_n_steps: int = 1000  # New: Frequency for periodic checkpointing

    # --- Gumbel-Softmax Configuration ---
    # These control the differentiable sampling of discrete tokens.
    gumbel_temperature: float = 1.0
    gumbel_anneal_rate: float = 1e-5  # Multiplicative decay rate per step
    gumbel_temperature_min: float = 0.1
    use_straight_through: bool = (
        True  # Use straight-through estimator for one-hot samples
    )

    # --- Dataset and Image Configuration ---
    image_size: int = 224  # Input image size (must match CLIP's expected input)
    image_channels: int = 3
    dataset_path: str = "~/datasets"

    # --- Operational Flags ---
    freeze_vision_encoder: bool = False  # Freeze the CLIP vision encoder
    freeze_llm_prior: bool = True  # Freeze the GPT-2 language model prior
    freeze_diffusion_decoder: bool = True  # Freeze the Stable Diffusion U-Net and VAE

    def __post_init__(self):
        """Log the configuration after initialization."""
        logger.info("--- VAE Configuration ---")
        for key, value in self.__dict__.items():
            logger.info(f"{key}: {value}")
        logger.info("-------------------------")
