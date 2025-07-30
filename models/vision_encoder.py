# models/vision_encoder.py

"""Vision Transformer encoder (patchify -> transformer)."""
from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from config import VAEConfig

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """
    Projects and reshapes image patches into a sequence of embeddings.

    Args:
        cfg (VAEConfig): Configuration object containing image and patch size.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        # Ensure image size is divisible by patch size
        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("Image size must be divisible by patch size.")
        self.patch_size = cfg.patch_size

        # A single convolution with kernel size and stride equal to the patch size
        # effectively splits the image into patches and embeds them.
        self.proj = nn.Conv2d(
            in_channels=cfg.image_channels,
            out_channels=cfg.encoder_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # Shape: [B, D, H/P, W/P]
        # Flatten the spatial dimensions into a single sequence dimension.
        # "b d h w -> b (h w) d"
        x = rearrange(x, "b d h w -> b (h w) d")
        logger.debug(f"PatchEmbedding output shape: {x.shape}")
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for both self-attention and cross-attention.

    Args:
        dim (int): Input and output dimension of tokens.
        heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        # --- IMPROVEMENT RECOMMENDATION ---
        # The following manual implementation can be replaced by `torch.nn.functional.scaled_dot_product_attention`
        # in PyTorch 2.0+ for significant performance and memory improvements. It fuses the operations into
        # a single kernel, which is much faster.
        # Example: F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout)

        # Separate linear layers for query, key, and value are clearer, especially for cross-attention.
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input sequence (query). Shape: [B, N, D].
            context (torch.Tensor, optional): The context sequence (key, value).
                                              If None, performs self-attention. Shape: [B, M, D].
            mask (torch.Tensor, optional): Attention mask.
        """
        B, N, D = x.shape
        # Use x as context for self-attention if no context is provided.
        kv_input = context if context is not None else x

        # --- BUG FIX ---
        # The original code concatenated x and context then did a single qkv projection,
        # which is incorrect for cross-attention where query, key, and value come from different sources.
        # This corrected version projects them separately.
        q = self.to_q(x)  # Query comes from the input sequence `x`
        k = self.to_k(
            kv_input
        )  # Key comes from the `context` (or `x` in self-attention)
        v = self.to_v(
            kv_input
        )  # Value comes from the `context` (or `x` in self-attention)

        # Reshape for multi-head attention: [B, N, D] -> [B, H, N, D/H]
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.heads)

        # Scaled Dot-Product Attention
        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        if mask is not None:
            # The mask should be broadcastable to the attention scores shape [B, H, N, M]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        output = torch.einsum("b h i j, b h j d -> b h i d", attn_probs, v)

        # Concatenate heads and project out
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    """
    A single block of a Transformer.

    Args:
        cfg (VAEConfig): Configuration object.
        cross_attention (bool): If True, adds a cross-attention layer.
    """

    def __init__(self, cfg: VAEConfig, cross_attention: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.encoder_dim)
        self.attn = MultiHeadAttention(
            cfg.encoder_dim, cfg.encoder_heads, cfg.encoder_dropout
        )

        self.cross_attention = cross_attention
        if self.cross_attention:
            self.norm_ctx = nn.LayerNorm(cfg.encoder_dim)
            self.cross_attn = MultiHeadAttention(
                cfg.encoder_dim, cfg.encoder_heads, cfg.encoder_dropout
            )

        self.norm2 = nn.LayerNorm(cfg.encoder_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.encoder_mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.encoder_mlp_dim, cfg.encoder_dim),
            nn.Dropout(cfg.encoder_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention part
        x = x + self.attn(self.norm1(x), mask=mask)

        # Cross-attention part (if applicable)
        if self.cross_attention:
            if context is None:
                raise ValueError("Context must be provided for cross-attention.")
            x = x + self.cross_attn(self.norm_ctx(x), context=context)

        # Feed-forward part
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) model to encode an image into a sequence of features.

    Args:
        cfg (VAEConfig): Configuration object.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.patch = PatchEmbedding(cfg)
        num_patches = (cfg.image_size // cfg.patch_size) ** 2

        # Special learnable token prepended to the sequence of patches.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_dim))
        # Learnable positional embeddings for each patch and the CLS token.
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.encoder_dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.encoder_depth)]
        )
        self.norm = nn.LayerNorm(cfg.encoder_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        logger.info(f"VisionEncoder initialized with {cfg.encoder_depth} blocks.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        logger.debug(f"VisionEncoder input shape: {x.shape}")

        # 1. Patchify and embed the image.
        x = self.patch(x)

        # 2. Prepend the CLS token to the patch sequence.
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # 3. Add positional embeddings.
        x = x + self.pos_emb

        # 4. Pass through Transformer blocks.
        for blk in self.blocks:
            x = blk(x)

        # 5. Final normalization.
        x = self.norm(x)
        logger.debug(f"VisionEncoder output shape: {x.shape}")
        return x
