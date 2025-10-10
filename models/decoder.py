from torch import nn
from einops import rearrange
import math
import config


class ViTDecoder(nn.Module):
    """
    The ViT Decoder for direct image reconstruction.
    This takes a sequence of latent codes and reconstructs an image using a
    Transformer followed by a convolutional head to stitch patches together.
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
        """Dynamically builds the convolutional head based on patch size."""
        layers = []
        # Calculate how many times we need to double the resolution
        num_upsamples = int(math.log2(config.PATCH_SIZE))

        in_channels = config.EMBEDDING_DIM
        for i in range(num_upsamples):
            out_channels = in_channels // 2
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            )
            layers.append(nn.ReLU())
            in_channels = out_channels

        # Final layer to match the original number of image channels
        layers.append(
            nn.Conv2d(in_channels, config.IN_CHANNELS, kernel_size=3, padding=1)
        )
        layers.append(nn.Tanh())  # Output pixels in [-1, 1] range

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The sequence of quantized codes, shape (B, num_patches, D).
        """
        # Process through Transformer blocks
        reconstructed_patches_embed = self.transformer_decoder(x)

        # Reshape sequence into a spatial feature map for the conv head
        h_w = int(config.NUM_PATCHES**0.5)
        feature_map = rearrange(
            reconstructed_patches_embed, "b (h w) c -> b c h w", h=h_w, w=h_w
        )

        # Reconstruct the final image
        reconstructed_image = self.conv_head(feature_map)
        return reconstructed_image
