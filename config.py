# config.py
from dataclasses import dataclass
from enum import Enum, auto


class ModelChoice(Enum):
    """Defines the available model architectures."""

    LONG_CONTEXT = auto()
    CONTROLNET = auto()
    HYBRID = auto()


@dataclass(frozen=True)
class SystemConfig:
    seed: int = 42
    output_dir: str = "./output"
    mixed_precision: str = "fp16"  # "no", "fp16", or "bf16"


@dataclass(frozen=True)
class DataConfig:
    dataset_path: str = "~/datasets"
    image_resolution: int = 512


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for all model components."""

    # --- CHOOSE YOUR MODEL ARCHITECTURE HERE ---
    active_model: ModelChoice = ModelChoice.HYBRID
    # -----------------------------------------

    # --- Component Model IDs ---
    # This is the VLM we will fine-tune during the training step.
    # It MUST be a vision-language model.
    training_vlm_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"

    # This is the captioner used for generating text at inference time.
    image_captioner_id: str = "Salesforce/blip2-opt-2.7b"

    # The base diffusion model.
    t2i_id: str = "runwayml/stable-diffusion-v1-5"

    # The ControlNet model.
    controlnet_id: str = "lllyasviel/sd-controlnet-canny"

    # This text-only LLM is not used in the current 'Codec' but is kept for future use.
    text_segmenter_id: str = "mistralai/Mistral-7B-Instruct-v0.2"


@dataclass(frozen=True)
class TrainingConfig:
    num_train_epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 4
    checkpointing_steps: int = 1000


@dataclass(frozen=True)
class AppConfig:
    """Root configuration for the application."""

    system: SystemConfig = SystemConfig()
    data: DataConfig = DataConfig()
    models: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()


# Create a single config instance for easy import
config = AppConfig()
