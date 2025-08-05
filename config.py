# config.py

"""
Centralized configuration for the End-to-End Swin-based Text-Conditioned VAE.
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
    """Configuration for the End-to-End Swin-based VAE."""

    # --- Model Architecture ---
    vision_encoder_model: str = "microsoft/swin-base-patch4-window7-224"
    llm_prior_model: str = "gpt2"

    # --- Text Decoder Specifics ---
    text_decoder_dim: int = 768
    text_decoder_heads: int = 12
    text_decoder_depth: int = 12
    text_decoder_mlp_dim: int = 3072
    max_text_length: int = 77

    # --- Custom Diffusion Decoder ---
    diffusion_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # --- Training Hyperparameters ---
    epochs: int = 150
    batch_size: int = 4
    learning_rate: float = 5e-5
    kl_weight: float = 1e-4
    seed: int = 42
    use_amp: bool = True
    work_dir: str = "./runs_swin_end_to_end"
    save_every_n_steps: int = 1000

    # --- Gumbel-Softmax Configuration ---
    gumbel_temperature: float = 1.0
    gumbel_anneal_rate: float = 1e-6
    gumbel_temperature_min: float = 0.1
    use_straight_through: bool = True

    # --- Dataset and Image Configuration ---
    image_size: int = 224
    image_channels: int = 3
    dataset_path: str = "~/datasets"

    # --- Operational Flags ---
    freeze_vision_encoder: bool = False
    freeze_llm_prior: bool = True
    freeze_diffusion_decoder: bool = False

    def __post_init__(self):
        """Log the configuration after initialization."""
        logger.info("--- VAE Swin End-to-End Configuration ---")
        for key, value in self.__dict__.items():
            logger.info(f"{key}: {value}")
        logger.info("-----------------------------------------")
