# models/swin_encoder.py

"""Swin Transformer V2 Vision Encoder (hierarchical transformer)."""
from __future__ import annotations
import torch
import torch.nn as nn
import logging
from .swin_transformer_v2 import SwinTransformerV2
from config import VAEConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SwinVisionEncoder(nn.Module):
    """
    Wrapper for the Swin Transformer V2 model to act as a vision encoder.

    Args:
        cfg (VAEConfig): Configuration with Swin-related hyperparameters.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # Initialize the SwinTransformerV2 model with parameters from the config
        self.model = SwinTransformerV2(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.image_channels,
            num_classes=0,  # We don't need the classification head
            embed_dim=cfg.encoder_dim,
            depths=cfg.swin_depths,
            num_heads=cfg.swin_num_heads,
            window_size=cfg.swin_window_size,
            mlp_ratio=cfg.swin_mlp_ratio,
            drop_path_rate=cfg.swin_drop_path_rate,
            # We want the raw feature sequence, not the final pooled output
            # so we will call forward_features instead of the full forward
        )

        # The final output dimension of the Swin encoder
        self.output_dim = self.model.num_features

        # This is a bit of a hack. The text decoder expects a specific dimension.
        # If the Swin output dim doesn't match, we add a projection layer.
        if self.output_dim != cfg.text_decoder_dim:
            self.feature_projection = nn.Linear(self.output_dim, cfg.text_decoder_dim)
            logger.info(
                f"Projecting Swin output from {self.output_dim} to {cfg.text_decoder_dim}"
            )
        else:
            self.feature_projection = nn.Identity()

        logger.info(f"SwinVisionEncoder initialized with depths: {cfg.swin_depths}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the input image through the Swin Transformer to get feature embeddings.

        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Tensor of shape [B, N, D] where D matches text_decoder_dim
        """
        # The SwinTransformerV2 class has a `forward_features` method
        # that returns the sequence of patch embeddings before the final pooling and head.
        # This gives us the feature sequence we need.
        features = self.model.forward_features(x)

        # Project features to the dimension expected by the text decoder
        projected_features = self.feature_projection(features)

        logger.debug(f"SwinVisionEncoder output shape: {projected_features.shape}")
        return projected_features
