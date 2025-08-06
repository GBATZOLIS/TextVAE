# config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    """Configuration for system-level settings."""

    seed: int = 42
    output_dir: str = "./output/semantic_codec_peft"
    mixed_precision: str = "fp16"  # "no", "fp16", or "bf16"


@dataclass(frozen=True)
class DataConfig:
    """Configuration for the dataset."""

    dataset_path: str = "~/datasets"  # Path will be expanded
    image_resolution: int = 1024


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the model components."""

    # The powerful VLM to use as the encoder
    vlm_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"

    # The frozen Text-to-Image model to use as the decoder
    t2i_id: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # The ControlNet for structural guidance
    controlnet_id: str = "diffusers/controlnet-canny-sdxl-1.0"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for the training process."""

    num_train_epochs: int = 10
    batch_size: int = 1  # Keep low due to high VRAM usage of VLM and SDXL
    learning_rate: float = 5e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2

    # Weight for the KL divergence loss to regularize the VLM's output
    kl_weight: float = 0.01

    # Temperature for Gumbel-Softmax sampling. A higher value encourages exploration.
    gumbel_temperature: float = 1.0


@dataclass(frozen=True)
class AppConfig:
    """Root configuration for the application."""

    system: SystemConfig = SystemConfig()
    data: DataConfig = DataConfig()
    models: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


# Create a single config instance for easy import
config = AppConfig()
