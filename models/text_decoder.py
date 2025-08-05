# models/text_decoder.py

"""
Autoregressive text decoder with cross-attention to vision features.
This version is designed to accept a shared token embedding layer from a
pretrained language model like GPT-2.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import VAEConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DecoderAttention(nn.Module):
    """A robust Multi-Head Attention module for the TextDecoder."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(
                f"Embedding dimension {dim} must be divisible by number of heads {heads}"
            )
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:

        kv_input = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # Reshape for multi-head attention
        q = q.view(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.heads, -1).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.heads, -1).transpose(1, 2)

        # Use PyTorch's optimized attention implementation
        # The is_causal flag handles the masking for autoregressive decoding
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Combine heads and project back to original dimension
        out = out.transpose(1, 2).reshape(x.shape)
        return self.to_out(out)


class DecoderBlock(nn.Module):
    """A single Transformer block for the TextDecoder."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.text_decoder_dim)
        self.attn1 = DecoderAttention(cfg.text_decoder_dim, cfg.text_decoder_heads)

        self.norm2 = nn.LayerNorm(cfg.text_decoder_dim)
        self.attn2 = DecoderAttention(cfg.text_decoder_dim, cfg.text_decoder_heads)

        self.norm3 = nn.LayerNorm(cfg.text_decoder_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.text_decoder_dim, cfg.text_decoder_mlp_dim),
            nn.GELU(),
            nn.Linear(cfg.text_decoder_mlp_dim, cfg.text_decoder_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:

        # Causal self-attention
        x = x + self.attn1(self.norm1(x), is_causal=True)

        # Cross-attention to image features
        x = x + self.attn2(self.norm2(x), context=context)

        # Feed-forward network
        x = x + self.mlp(self.norm3(x))

        return x


class TextDecoder(nn.Module):
    """
    Autoregressive Transformer decoder to generate text from image features.
    """

    def __init__(
        self, cfg: VAEConfig, token_embedding_layer: nn.Embedding, vocab_size: int
    ):
        super().__init__()
        self.cfg = cfg

        # FIX: Correctly accept and use the shared token embedding layer
        self.token_emb = token_embedding_layer

        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg.max_text_length, cfg.text_decoder_dim)
        )

        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.text_decoder_depth)]
        )

        self.norm = nn.LayerNorm(cfg.text_decoder_dim)

        # The final head projects back to the vocabulary size.
        self.lm_head = nn.Linear(cfg.text_decoder_dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        logger.info(f"TextDecoder initialized with {cfg.text_decoder_depth} blocks.")

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:  # Return type updated for clarity

        B, T = tokens.shape

        if T > self.cfg.max_text_length:
            raise ValueError(
                f"Input sequence length {T} exceeds maximum {self.cfg.max_text_length}"
            )

        # Embed tokens and add positional embeddings
        x = self.token_emb(tokens)
        x = x + self.pos_emb[:, :T]

        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, context=image_feats)

        x = self.norm(x)
        logits = self.lm_head(x)

        # KV caching is omitted for simplicity in this robust version,
        # but can be added back for optimized inference if needed.
        return logits, None
