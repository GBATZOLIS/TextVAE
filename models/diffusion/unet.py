# models/diffusion/unet.py

"""U-Net with cross-attention to text embeddings for DDPM."""
from __future__ import annotations
import math
import logging
import torch
import torch.nn as nn
from config import VAEConfig

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Creates sinusoidal position embeddings for the diffusion timesteps.
    This allows the model to know at which timestep `t` it is operating.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, but got {dim}")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps (torch.Tensor): A 1D tensor of timesteps. Shape: [B].

        Returns:
            torch.Tensor: The computed position embeddings. Shape: [B, dim].
        """
        device = timesteps.device
        half_dim = self.dim // 2

        # 1. Calculate the scaling factor for the frequencies (this is a float)
        freq_exponent = math.log(10000) / (half_dim - 1)

        # 2. Create the inverse frequencies (this is a 1D Tensor)
        inv_freq = torch.exp(torch.arange(half_dim, device=device) * -freq_exponent)

        # 3. Calculate the arguments for the sin/cos functions (a 2D Tensor)
        #    timesteps[:, None] -> [B, 1]
        #    inv_freq[None, :]  -> [1, half_dim]
        pos_enc_arg = timesteps[:, None] * inv_freq[None, :]

        # 4. Concatenate sin and cos components to form the final embedding
        embedding = torch.cat([pos_enc_arg.sin(), pos_enc_arg.cos()], dim=-1)

        logger.debug(
            f"SinusoidalPositionEmbeddings created with shape: {embedding.shape}"
        )
        return embedding


class ResidualBlock(nn.Module):
    """
    A residual block with a time embedding MLP and cross-attention to text features.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        t_dim (int): Dimension of the time embedding.
        cond_dim (int): Dimension of the text conditioning embedding.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # MLP to project time embedding to the same dimension as feature maps
        self.time_mlp = nn.Linear(t_dim, out_ch)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Cross-attention layers
        self.text_proj = nn.Linear(cond_dim, out_ch)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_ch, num_heads=8, batch_first=True
        )

        self.nonlinear = nn.SiLU()
        # Shortcut connection to match output channels if they differ from input
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, txt: torch.Tensor
    ) -> torch.Tensor:
        h = self.nonlinear(self.norm1(x))
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]  # Reshape t_emb for broadcasting

        h = self.nonlinear(self.norm2(h))
        h = self.conv2(h)

        # --- Cross-Attention ---
        B, C, H, W = h.shape
        # Flatten spatial dimensions to a sequence for attention
        h_flat = h.view(B, C, H * W).transpose(1, 2)  # Shape: [B, H*W, C]
        # Project text conditioning to the right dimension
        txt_proj = self.text_proj(txt)  # Shape: [B, SeqLen, C]

        # Attend: query is the image, key/value is the text
        h_attn, _ = self.attn(query=h_flat, key=txt_proj, value=txt_proj)

        # Add attention output back to the feature map
        h = h + h_attn.transpose(1, 2).view(B, C, H, W)

        # Add residual connection
        return h + self.shortcut(x)


class UNet(nn.Module):
    """
    A U-Net model with residual blocks, attention, and conditioning.

    This network takes a noised image, a timestep, and text conditioning,
    and outputs the predicted noise.

    Args:
        cfg (VAEConfig): Configuration object.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        ch = 64  # Base channel count
        time_dim = 256
        t_mlp_dim = time_dim * 4

        # --- Time Embedding Projection ---
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, t_mlp_dim),
            nn.SiLU(),
            nn.Linear(t_mlp_dim, t_mlp_dim),
        )

        # --- Downsampling Path (Encoder) ---
        self.init_conv = nn.Conv2d(cfg.image_channels, ch, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(ch, ch * 2, t_mlp_dim, cfg.text_decoder_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResidualBlock(ch * 2, ch * 4, t_mlp_dim, cfg.text_decoder_dim)
        self.pool2 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        self.mid = ResidualBlock(ch * 4, ch * 4, t_mlp_dim, cfg.text_decoder_dim)

        # --- Upsampling Path (Decoder) ---
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 4, kernel_size=2, stride=2)
        # Skip connection is concatenated, so input channels are doubled
        self.dec2 = ResidualBlock(
            ch * 4 + ch * 4, ch * 2, t_mlp_dim, cfg.text_decoder_dim
        )

        self.up1 = nn.ConvTranspose2d(ch * 2, ch * 2, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(ch * 2 + ch * 2, ch, t_mlp_dim, cfg.text_decoder_dim)

        # --- Final Output Layer ---
        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, cfg.image_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, txt: torch.Tensor
    ) -> torch.Tensor:
        logger.debug(
            f"UNet input shapes - x: {x.shape}, t: {t.shape}, txt: {txt.shape}"
        )
        # 1. Get time embeddings
        t_emb = self.time_mlp(t)

        # 2. Downsampling path
        h0 = self.init_conv(x)
        h1 = self.down1(h0, t_emb, txt)
        p1 = self.pool1(h1)
        h2 = self.down2(p1, t_emb, txt)
        p2 = self.pool2(h2)

        # 3. Bottleneck
        h_mid = self.mid(p2, t_emb, txt)

        # 4. Upsampling path with skip connections
        up2 = self.up2(h_mid)
        # Concatenate skip connection from downsampling path
        cat2 = torch.cat([up2, h2], dim=1)
        dec2 = self.dec2(cat2, t_emb, txt)

        up1 = self.up1(dec2)
        cat1 = torch.cat([up1, h1], dim=1)
        dec1 = self.dec1(cat1, t_emb, txt)

        # 5. Final output
        out = self.final(dec1)
        logger.debug(f"UNet output shape: {out.shape}")
        return out
