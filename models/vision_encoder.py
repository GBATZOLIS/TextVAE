"""Vision Transformer encoder (patchify -> transformer)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from config import VAEConfig

class PatchEmbedding(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(
            cfg.image_channels, cfg.encoder_dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return rearrange(x, "b d h w -> b (h w) d")

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        if context is None:
            context = x
        qkv = self.qkv(torch.cat([x, context], dim=1))
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, self.heads, -1).transpose(1, 2)
        k = k.view(B, -1, self.heads, D // self.heads).transpose(1, 2)
        v = v.view(B, -1, self.heads, D // self.heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: VAEConfig, cross_attention: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.encoder_dim)
        self.attn = MultiHeadAttention(cfg.encoder_dim, cfg.encoder_heads, cfg.encoder_dropout)
        self.cross_attention = cross_attention
        if cross_attention:
            self.norm_ctx = nn.LayerNorm(cfg.encoder_dim)
            self.cross_attn = MultiHeadAttention(cfg.encoder_dim, cfg.encoder_heads, cfg.encoder_dropout)
        self.norm2 = nn.LayerNorm(cfg.encoder_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.encoder_mlp_dim),
            nn.GELU(),
            nn.Dropout(cfg.encoder_dropout),
            nn.Linear(cfg.encoder_mlp_dim, cfg.encoder_dim),
            nn.Dropout(cfg.encoder_dropout),
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x))
        if self.cross_attention and context is not None:
            x = x + self.cross_attn(self.norm_ctx(x), context=context)
        x = x + self.mlp(self.norm2(x))
        return x

class VisionEncoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.patch = PatchEmbedding(cfg)
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_dim))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.encoder_dim))
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.encoder_depth)])
        self.norm = nn.LayerNorm(cfg.encoder_dim)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
