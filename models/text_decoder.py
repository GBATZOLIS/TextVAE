# models/text_decoder.py

"""
Autoregressive text decoder with cross-attention to vision features.
This version is adapted to work with features from a CLIP vision encoder
and includes placeholders for KV caching for efficient inference.
"""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import VAEConfig

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DecoderAttention(nn.Module):
    """
    A more robust Multi-Head Attention module for the TextDecoder.
    It uses PyTorch 2.0's fused `scaled_dot_product_attention` for performance.
    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        kv_input = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # Handle KV caching for fast autoregressive generation
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat((past_k, k), dim=1)
            v = torch.cat((past_v, v), dim=1)

        # Use PyTorch's optimized attention implementation
        out = (
            F.scaled_dot_product_attention(
                q.view(q.shape[0], q.shape[1], self.heads, -1).transpose(1, 2),
                k.view(k.shape[0], k.shape[1], self.heads, -1).transpose(1, 2),
                v.view(v.shape[0], v.shape[1], self.heads, -1).transpose(1, 2),
                is_causal=is_causal,
            )
            .transpose(1, 2)
            .reshape(q.shape)
        )

        return self.to_out(out), (k, v)


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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # Self-attention (causal)
        attn_out, self_kv = self.attn1(self.norm1(x), is_causal=True, kv_cache=kv_cache)
        x = x + attn_out

        # Cross-attention to image features
        cross_out, _ = self.attn2(self.norm2(x), context=context)
        x = x + cross_out

        # Feed-forward
        x = x + self.mlp(self.norm3(x))

        return x, self_kv


class TextDecoder(nn.Module):
    """
    Autoregressive Transformer decoder to generate text from image features.
    """

    def __init__(
        self, cfg: VAEConfig, token_embedding_layer: nn.Embedding, vocab_size: int
    ):
        super().__init__()
        self.cfg = cfg
        # Share token embeddings with the LLM prior for a consistent space
        self.token_emb = token_embedding_layer

        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg.max_text_length, cfg.text_decoder_dim)
        )

        self.blocks = nn.ModuleList(
            [DecoderBlock(cfg) for _ in range(cfg.text_decoder_depth)]
        )

        self.norm = nn.LayerNorm(cfg.text_decoder_dim)
        # The final head projects back to the vocabulary size.
        # FIX: Use the vocab_size passed in from the loaded model, not from the config string.
        self.lm_head = nn.Linear(cfg.text_decoder_dim, vocab_size, bias=False)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        logger.info(f"TextDecoder initialized with {cfg.text_decoder_depth} blocks.")

    def forward(
        self,
        tokens: torch.Tensor,
        image_feats: torch.Tensor,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:

        B, T = tokens.shape

        if T > self.cfg.max_text_length:
            raise ValueError(
                f"Sequence length {T} exceeds max {self.cfg.max_text_length}"
            )

        x = self.token_emb(tokens)
        x = x + self.pos_emb[:, :T]

        new_kv_caches = []
        for i, blk in enumerate(self.blocks):
            # Pass the corresponding cache to each block
            kv_cache = past_key_values[i] if use_cache and past_key_values else None
            x, new_kv = blk(x, context=image_feats, kv_cache=kv_cache)
            if use_cache:
                new_kv_caches.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_kv_caches
