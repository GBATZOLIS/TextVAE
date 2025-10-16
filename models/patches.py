# models/patches.py

import torch
from torch import nn
from einops.layers.torch import Rearrange
import config


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.
    """

    def __init__(self):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=config.IN_CHANNELS,
                out_channels=config.EMBEDDING_DIM,
                kernel_size=config.PATCH_SIZE,
                stride=config.PATCH_SIZE,
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.positional_embedding = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES, config.EMBEDDING_DIM)
        )

    def forward(self, x):
        patches = self.patcher(x)
        patches += self.positional_embedding
        return patches
