# models/swin_encoder.py

"""Swin Transformer V2 Vision Encoder (hierarchical transformer)."""
from __future__ import annotations
import torch
import torch.nn as nn
import logging
from swin_transformer_v2 import PatchEmbed, BasicLayer
from config import VAEConfig  # Your config object

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SwinVisionEncoder(nn.Module):
    """
    Swin Transformer V2-based vision encoder.

    Args:
        cfg (VAEConfig): Configuration with Swin-related hyperparameters.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # Patch embedding to convert image to patch tokens
        self.patch_embed = PatchEmbed(
            img_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.image_channels,
            embed_dim=cfg.encoder_dim,
            norm_layer=nn.LayerNorm,
        )

        # Stages of Swin Transformer blocks (BasicLayer)
        self.stages = nn.ModuleList()
        embed_dim = cfg.encoder_dim
        for i_layer in range(len(cfg.depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                ),
                depth=cfg.depths[i_layer],
                num_heads=cfg.num_heads[i_layer],
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
                drop=cfg.dropout,
                downsample=True if i_layer < len(cfg.depths) - 1 else False,
            )
            self.stages.append(layer)

        # Final layer normalization
        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (len(cfg.depths) - 1)))

        logger.info(f"SwinVisionEncoder initialized with {len(cfg.depths)} stages.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Tensor of shape [B, N, D] where N is the final sequence length and D is the final embedding dim
        """
        x = self.patch_embed(x)  # B, C, H', W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).view(B, H, W, C)  # B, H, W, C

        for stage in self.stages:
            x = stage(x)

        x = x.view(B, -1, x.shape[-1])  # flatten spatial dims
        x = self.norm(x)  # B, N, D
        return x
