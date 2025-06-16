"""Autoregressive text decoder with cross‑attention to vision features."""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from config import VAEConfig
from .vision_encoder import MultiHeadAttention, TransformerBlock

class TextDecoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.text_decoder_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_text_length, cfg.text_decoder_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, cross_attention=True) for _ in range(cfg.text_decoder_depth)]
        )
        self.norm = nn.LayerNorm(cfg.text_decoder_dim)
        self.lm_head = nn.Linear(cfg.text_decoder_dim, cfg.vocab_size, bias=False)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()[None, None]

    def forward(self, tokens: torch.Tensor, image_feats: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        B, T = tokens.shape[:2]
        if tokens.dim() == 2:  # hard tokens
            x = self.token_emb(tokens)
        else:  # soft tokens
            x = torch.matmul(tokens, self.token_emb.weight)
        x = x + self.pos_emb[:, :T]
        mask = self.causal_mask(T, tokens.device)
        for blk in self.blocks:
            x = blk(x, context=image_feats, mask=mask)
        x = self.norm(x)
        return self.lm_head(x)
