from torch import nn
import config


class ViTEncoder(nn.Module):
    """
    The Vision Transformer (ViT) Encoder for the VQ-VAE.

    This module takes a sequence of patch embeddings and processes them through
    a series of Transformer blocks. It outputs a feature map ready for quantization.
    """

    def __init__(self):
        super().__init__()
        # Standard Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.ENCODER_HEADS,
            dim_feedforward=config.EMBEDDING_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )

        # Stack of Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.ENCODER_LAYERS
        )

    def forward(self, x):
        """
        Forward pass for the ViTEncoder.
        Args:
            x (torch.Tensor): Input sequence of patch embeddings,
                          shape (B, num_patches, embedding_dim).
        Returns:
            torch.Tensor: The encoded feature map for quantization,
                          shape (B, num_patches, embedding_dim).
        """
        # Pass through the transformer blocks
        encoded_patches = self.transformer_encoder(x)  # (B, 64, 256)
        return encoded_patches
