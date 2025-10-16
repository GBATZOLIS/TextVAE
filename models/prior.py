# models/prior.py

import torch
from torch import nn
import config


class CodebookPrior(nn.Module):
    """
    An autoregressive Transformer model to learn the prior distribution
    of the VQ-VAE codebook indices.
    """

    def __init__(self, num_codes, embedding_dim, nhead, num_layers, dropout):
        super().__init__()
        self.num_codes = num_codes

        # --- Components ---
        # An embedding layer for the codebook indices + a Start-Of-Sequence token
        self.token_embedding = nn.Embedding(num_codes + 1, embedding_dim)

        # Standard positional encoding
        self.positional_embedding = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES + 1, embedding_dim)
        )

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        # The Transformer Decoder itself
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # A final linear layer to project to the codebook vocabulary size
        self.output_head = nn.Linear(embedding_dim, num_codes)

        # A special token to signify the start of a sequence
        self.sos_token = nn.Parameter(
            torch.randn(1, 1, embedding_dim), requires_grad=False
        )

    def forward(self, indices):
        """
        Forward pass for training the prior.
        Args:
            indices (torch.Tensor): A batch of ground-truth codebook indices.
                                    Shape: (batch_size, num_patches).
        Returns:
            torch.Tensor: Logits over the codebook vocabulary for each position.
        """
        # Embed the input indices
        token_embeds = self.token_embedding(indices)
        b, n, d = token_embeds.shape

        # Prepend the Start-Of-Sequence (SOS) token embedding
        sos_embed = self.sos_token.expand(b, -1, -1)
        seq = torch.cat([sos_embed, token_embeds], dim=1)[:, :-1, :]  # Shift right

        # Add positional embeddings
        seq += self.positional_embedding[:, :n, :]

        # Generate a causal mask to prevent attention to future tokens
        causal_mask = self.generate_square_subsequent_mask(n, device=indices.device)

        # Pass through the transformer decoder (using itself as memory)
        output = self.transformer_decoder(
            tgt=seq, memory=seq, tgt_mask=causal_mask, memory_mask=causal_mask
        )

        # Project to logits
        logits = self.output_head(output)

        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        """Generates a square causal mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
