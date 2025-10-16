import torch
from torch import nn
import config


class Prior(nn.Module):
    """
    An autoregressive Transformer model to learn the distribution of latent codes.

    This model takes a sequence of latent code indices and learns to predict
    the next code in the sequence. It's the "grammar" engine for the VQ-VAE.
    """

    def __init__(self):
        super().__init__()
        # An embedding layer for the discrete latent codes from the VQ-VAE codebook
        self.code_embedding = nn.Embedding(config.NUM_EMBEDDINGS, config.EMBEDDING_DIM)

        # A positional embedding for the sequence of codes
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES, config.EMBEDDING_DIM)
        )

        # A standard Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.ENCODER_HEADS,  # Using same head count as VQ-VAE for consistency
            dim_feedforward=config.EMBEDDING_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.ENCODER_LAYERS,  # Using same layer count as VQ-VAE
        )

        # The output layer projects the transformer's output to logits over the codebook
        self.output_projection = nn.Linear(config.EMBEDDING_DIM, config.NUM_EMBEDDINGS)

    def forward(self, x):
        """
        Forward pass for the Prior model.
        Args:
            x (torch.Tensor): A sequence of latent code indices.
                              Shape: (B, sequence_length).
        Returns:
            torch.Tensor: The logits for predicting the next code in the sequence.
                          Shape: (B, sequence_length, num_embeddings).
        """
        # Embed the input code indices and add positional information
        x = self.code_embedding(x) + self.pos_embedding[:, : x.size(1), :]

        # Process the sequence through the transformer
        x = self.transformer(x)

        # Project to logits
        logits = self.output_projection(x)

        return logits
