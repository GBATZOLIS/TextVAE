# models/decoder.py

from torch import nn
from einops import rearrange
import math
import config


class ViTDecoder(nn.Module):
    """
    The ViT Decoder for direct image reconstruction.
    """

    def __init__(self):
        super().__init__()
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.EMBEDDING_DIM,
            nhead=config.DECODER_HEADS,
            dim_feedforward=config.EMBEDDING_DIM * 4,
            dropout=config.DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=config.DECODER_LAYERS
        )
        self.conv_head = self._build_conv_head()

    def _build_conv_head(self):
        """Builds the convolutional head for upsampling patches to an image."""
        layers = []
        # Calculate the number of upsampling layers needed based on patch size
        num_upsamples = int(math.log2(config.PATCH_SIZE))
        in_channels = config.EMBEDDING_DIM

        # Add transpose convolution layers for upsampling
        for i in range(num_upsamples):
            out_channels = in_channels // 2
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Final layer to match the number of input channels
        layers.append(
            nn.Conv2d(in_channels, config.IN_CHANNELS, kernel_size=3, padding=1)
        )
        # Use Tanh activation as images are normalized to [-1, 1]
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    # --- MODIFIED: Accept the attention mask ---
    def forward(self, x, mask=None):
        """
        Forward pass for the decoder.
        Args:
            x (torch.Tensor): The sequence of quantized latent vectors.
            mask (torch.Tensor, optional): The attention mask to ignore truncated tokens.
                                           Shape: (batch_size, num_patches).
                                           Defaults to None.
        """
        # Pass the mask to the transformer decoder
        reconstructed_patches_embed = self.transformer_decoder(
            x, src_key_padding_mask=mask
        )

        # Reshape the sequence back into a feature map
        h_w = int(config.NUM_PATCHES**0.5)
        feature_map = rearrange(
            reconstructed_patches_embed, "b (h w) c -> b c h w", h=h_w, w=h_w
        )

        # Reconstruct the image using the convolutional head
        reconstructed_image = self.conv_head(feature_map)

        return reconstructed_image
