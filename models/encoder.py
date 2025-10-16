# models/encoder.py

from torch import nn
import config


class ViTEncoder(nn.Module):
    """
    The Vision Transformer (ViT) Encoder for the VQ-VAE.
    """

    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.ENCODER_HEADS,
            dim_feedforward=config.EMBEDDING_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.ENCODER_LAYERS
        )

    def forward(self, x):
        return self.transformer_encoder(x)
