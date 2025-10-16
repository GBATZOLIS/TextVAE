# models/decoder.py

from torch import nn
from einops import rearrange
import math
import config
import torch


class ViTDecoder(nn.Module):
    """
    A Transformer Decoder that generates a full image from a variable-length
    sequence of context tokens using cross-attention.
    """

    def __init__(self):
        super().__init__()
        # A TransformerDecoderLayer contains both self-attention and cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.DECODER_HEADS,
            dim_feedforward=config.EMBEDDING_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.DECODER_LAYERS
        )

        # --- NEW: Learnable query tokens ---
        # These act as the initial input to the decoder, representing the "slots"
        # for the N output patches that will form the final image.
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES, config.EMBEDDING_DIM)
        )

        # --- NEW: Decoder-specific positional embeddings ---
        # Added to the query tokens so the decoder knows the spatial location
        # of the patch it is generating.
        self.positional_embedding = nn.Parameter(
            torch.randn(1, config.NUM_PATCHES, config.EMBEDDING_DIM)
        )

        self.conv_head = self._build_conv_head()

    def _build_conv_head(self):
        layers = []
        num_upsamples = int(math.log2(config.PATCH_SIZE))
        in_channels = config.EMBEDDING_DIM
        for i in range(num_upsamples):
            out_channels = in_channels // 2
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(
            nn.Conv2d(in_channels, config.IN_CHANNELS, kernel_size=3, padding=1)
        )
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, context, key_padding_mask=None):
        """
        Forward pass for the ViTDecoder.
        Args:
            context (torch.Tensor): A variable-length sequence of quantized features (B, S, D)
                                    that provides the context for generation.
            key_padding_mask (torch.Tensor, optional): A mask for the context sequence of shape (B, S)
                                                       to indicate which elements are padding.
        """
        # Expand query tokens for the batch and add positional information
        queries = self.query_tokens.expand(context.shape[0], -1, -1)
        queries = queries + self.positional_embedding

        # The learnable queries (tgt) attend to the input context (memory)
        # to generate the full sequence of output patch embeddings.
        reconstructed_patches_embed = self.transformer_decoder(
            tgt=queries, memory=context, memory_key_padding_mask=key_padding_mask
        )

        h_w = int(config.NUM_PATCHES**0.5)
        feature_map = rearrange(
            reconstructed_patches_embed, "b (h w) c -> b c h w", h=h_w, w=h_w
        )
        reconstructed_image = self.conv_head(feature_map)
        return reconstructed_image
