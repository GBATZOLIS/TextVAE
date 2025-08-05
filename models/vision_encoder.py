# models/vision_encoder.py

"""
Vision Encoder using a pre-trained CLIP model.

This module replaces the from-scratch ViT with a powerful, pre-trained
vision transformer from OpenAI's CLIP. This provides a massive head start,
as the model already understands a rich variety of visual concepts in a
way that is aligned with language.
"""
from __future__ import annotations
import logging
import torch
from torch import nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from config import VAEConfig

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class CLIPVisionEncoder(nn.Module):
    """
    A wrapper for the CLIP vision model to be used as the VAE's vision encoder.

    It loads a pre-trained CLIP model and freezes its parameters to act as a
    static, powerful feature extractor.

    Args:
        cfg (VAEConfig): Configuration object.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        try:
            logger.info(f"Loading CLIP vision model: {cfg.vision_encoder_model}")
            self.vision_tower = CLIPVisionModel.from_pretrained(
                cfg.vision_encoder_model
            )
            self.image_processor = CLIPImageProcessor.from_pretrained(
                cfg.vision_encoder_model
            )
            logger.info("CLIP vision model loaded successfully.")
        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to load CLIP model '{cfg.vision_encoder_model}'. "
                "Check model name and internet connection."
            )
            raise e

        # --- Freeze Parameters ---
        # We use CLIP as a frozen feature extractor. Its weights are not updated
        # during training, which saves significant compute resources.
        if cfg.freeze_vision_encoder:
            logger.info("Freezing CLIP vision encoder parameters.")
            for param in self.vision_tower.parameters():
                param.requires_grad = False
            self.vision_tower.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of images into a sequence of feature embeddings.

        Args:
            images (torch.Tensor): A batch of images, assumed to be in the
                                   range [-1, 1]. Shape: [B, C, H, W].

        Returns:
            torch.Tensor: The image features (last hidden state from CLIP).
                          Shape: [B, num_patches, feature_dim].
        """
        # The CLIP model expects images normalized to the range [0, 1].
        # We convert from [-1, 1] to [0, 1] before processing.
        images_0_to_1 = (images + 1) / 2.0

        # The `CLIPImageProcessor` handles resizing, normalization, and formatting.
        # FIX: Set do_rescale=False to prevent the processor from rescaling images that are already in the [0, 1] range.
        inputs = self.image_processor(
            images=images_0_to_1, return_tensors="pt", do_rescale=False
        ).to(images.device)

        # Forward pass through the frozen vision tower.
        with torch.no_grad() if self.cfg.freeze_vision_encoder else torch.enable_grad():
            outputs = self.vision_tower(**inputs)
            # We use the last hidden state as the primary feature representation.
            # This provides a rich, sequential output of patch features.
            image_features = outputs.last_hidden_state

        logger.debug(f"CLIPVisionEncoder output shape: {image_features.shape}")
        return image_features
