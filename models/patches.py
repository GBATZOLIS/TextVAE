import torch
from torch import nn
from einops.layers.torch import Rearrange
import config


class PatchEmbedding(nn.Module):
    """
    Splits an image into patches and embeds them.

    This module takes an image tensor and performs the following steps:
    1.  Uses a convolutional layer to create patches and embed them in a single step.
        The convolution's kernel size and stride are equal to the patch size.
    2.  Flattens the spatial dimensions of the feature map into a sequence.
    3.  Adds a learnable positional embedding to each patch embedding to retain
        spatial information.
    """

    def __init__(self):
        super().__init__()
        # Calculate the dimensionality of each flattened patch
        # patch_dim = config.IN_CHANNELS * config.PATCH_SIZE * config.PATCH_SIZE

        # The core of patching and embedding: a single convolutional layer
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=config.IN_CHANNELS,
                out_channels=config.EMBEDDING_DIM,
                kernel_size=config.PATCH_SIZE,
                stride=config.PATCH_SIZE,
            ),
            # Rearrange to create a sequence of patches for the Transformer
            Rearrange("b c h w -> b (h w) c"),
        )

        # Learnable positional embeddings for each patch
        self.positional_embedding = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES, config.EMBEDDING_DIM)
        )

    def forward(self, x):
        """
        Forward pass for the PatchEmbedding module.
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: The sequence of patch embeddings with positional information,
                          shape (B, num_patches, embedding_dim).
        """
        # Create patch embeddings
        patches = self.patcher(x)
        # Add positional embeddings
        patches += self.positional_embedding
        return patches
