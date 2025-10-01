# models/text_decoder.py

"""Autoregressive text decoder with cross-attention to vision features."""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from config import VAEConfig
from .vision_encoder import TransformerBlock

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextDecoder(nn.Module):
    """
    An autoregressive Transformer-based text decoder.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.text_decoder_dim)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg.max_text_length, cfg.text_decoder_dim)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg, cross_attention=True)
                for _ in range(cfg.text_decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.text_decoder_dim)
        self.lm_head = nn.Linear(cfg.text_decoder_dim, cfg.vocab_size, bias=False)

        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        logger.info(f"TextDecoder initialized with {cfg.text_decoder_depth} blocks.")

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal mask to prevent attention to future tokens.
        """
        mask = torch.ones(seq_len, seq_len, device=device).tril_().bool()
        return mask[None, None, :, :]

    def forward(
        self, tokens: torch.Tensor, image_feats: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        B, T = tokens.shape[:2]

        if use_cache:
            logger.warning(
                "`use_cache=True` was passed, but KV caching is not implemented. "
                "Performance for autoregressive generation will be suboptimal."
            )

        if T > self.cfg.max_text_length:
            raise ValueError(
                f"Input sequence length ({T}) exceeds maximum configured length "
                f"({self.cfg.max_text_length})."
            )

        logger.debug(
            f"TextDecoder input shapes - tokens: {tokens.shape}, image_feats: {image_feats.shape}"
        )

        if tokens.dim() == 2:
            x = self.token_emb(tokens)
        elif tokens.shape[-1] == self.cfg.vocab_size:
            x = torch.matmul(tokens, self.token_emb.weight)
        else:
            x = tokens

        x = x + self.pos_emb[:, :T]
        mask = self.causal_mask(T, tokens.device)

        for blk in self.blocks:
            x = blk(x, context=image_feats, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        logger.debug(f"TextDecoder output logits shape: {logits.shape}")
        return logits
