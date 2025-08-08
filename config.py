# config.py

"""
Centralized configuration for the Semantic Compression project using dataclasses.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configurations for model architectures."""

    # vision_encoder_name: str = "microsoft/swin-base-patch4-window7-224-in22k"
    vision_encoder_name: str = "microsoft/swin-tiny-patch4-window7-224"

    vision_encoder_max_text_length: int = 128

    diffusion_decoder_unet_name: str = "runwayml/stable-diffusion-v1-5"
    diffusion_decoder_scheduler_type: str = "ddpm"
    diffusion_decoder_chunk_size: int = 75  # Corresponds to CLIP's token limit

    plausibility_module_name: str = "gpt2"


@dataclass
class TrainingConfig:
    """Configurations for the training process."""

    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1.0e-4
    lambda_plausibility: float = 0.01  # Weight for the KL penalty
    use_amp: bool = False  # Use Automatic Mixed Precision
    gradient_clip_val: float = 1.0


@dataclass
class DataConfig:
    """Configurations for data loading and processing."""

    data_dir: str = "~/datasets/celeba/img_align_celeba"
    image_size: int = 224


@dataclass
class LoggingConfig:
    """Configurations for logging and checkpointing."""

    run_name: str = "semantic_compressor_v2"
    project_name: str = "semantic-compression"
    use_wandb: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_every_n_epochs: int = 5
    save_every_n_steps: int = 5000


@dataclass
class ProjectConfig:
    """Root configuration class for the project."""

    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        """Log the complete configuration after initialization."""
        logger.info("--- Project Configuration Loaded ---")
        for section, section_config in self.__dict__.items():
            logger.info(f"[{section.upper()}]")
            for key, value in section_config.__dict__.items():
                logger.info(f"  {key}: {value}")
        logger.info("------------------------------------")

        # Create checkpoint directory if it doesn't exist
        Path(self.logging.checkpoint_dir).mkdir(exist_ok=True)
