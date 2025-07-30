# models/text_decoder.py

"""Autoregressive text decoder with cross-attention to vision features."""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from config import VAEConfig
from .vision_encoder import TransformerBlock

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextDecoder(nn.Module):
    """
    An autoregressive Transformer-based text decoder.

    It generates text tokens one by one, conditioned on image features via
    cross-attention.

    Args:
        cfg (VAEConfig): Configuration object with model parameters.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.text_decoder_dim)
        # Positional embeddings are learned parameters.
        self.pos_emb = nn.Parameter(
            torch.zeros(1, cfg.max_text_length, cfg.text_decoder_dim)
        )
        self.blocks = nn.ModuleList(
            # Each block has self-attention and cross-attention to image features.
            [
                TransformerBlock(cfg, cross_attention=True)
                for _ in range(cfg.text_decoder_depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.text_decoder_dim)
        # The final linear layer to project back to the vocabulary size.
        self.lm_head = nn.Linear(cfg.text_decoder_dim, cfg.vocab_size, bias=False)

        # Initialize positional embeddings.
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        logger.info(f"TextDecoder initialized with {cfg.text_decoder_depth} blocks.")

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a causal mask to prevent attention to future tokens.

        For a sequence of length `seq_len`, returns a `[seq_len, seq_len]`
        lower-triangular matrix of booleans.
        """
        # torch.tril creates the lower-triangular matrix.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        # Add batch and head dimensions for broadcasting: [1, 1, seq_len, seq_len]
        return mask[None, None, :, :]

    def forward(
        self, tokens: torch.Tensor, image_feats: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        """
        Forward pass of the text decoder.

        Args:
            tokens (torch.Tensor): Input text tokens (either hard or soft).
                                   Shape: [B, T] or [B, T, V].
            image_feats (torch.Tensor): Conditioning image features from the vision encoder.
                                        Shape: [B, N, D].
            use_cache (bool): If True, enables key-value caching for faster generation.
                              NOTE: This feature is not implemented in the original code.
                              Implementing it would significantly speed up autoregressive sampling.

        Returns:
            torch.Tensor: Logits over the vocabulary. Shape: [B, T, vocab_size].
        """
        B, T = tokens.shape[:2]

        # --- BUG/IMPROVEMENT NOTE ---
        if use_cache:
            logger.warning(
                "`use_cache=True` was passed, but KV caching is not implemented. "
                "Performance for autoregressive generation will be suboptimal."
            )

        # Ensure sequence length does not exceed positional embedding size.
        if T > self.cfg.max_text_length:
            raise ValueError(
                f"Input sequence length ({T}) exceeds maximum configured length "
                f"({self.cfg.max_text_length})."
            )

        logger.debug(
            f"TextDecoder input shapes - tokens: {tokens.shape}, image_feats: {image_feats.shape}"
        )

        # 1. Embed tokens.
        # The input 'tokens' can be one of three things. We need to handle each case.
        if tokens.dim() == 2:
            # Case 1: Hard tokens (indices) -> Embed them.
            x = self.token_emb(tokens)
        elif tokens.shape[-1] == self.cfg.vocab_size:
            # Case 2: Soft tokens (probabilities) -> Embed via matmul.
            x = torch.matmul(tokens, self.token_emb.weight)
        else:
            # Case 3: Already embedded tokens -> Use them directly.
            # This is the case during the autoregressive generation loop.
            x = tokens

        # 2. Add positional embeddings.
        x = x + self.pos_emb[:, :T]

        # 3. Create causal mask for self-attention.
        mask = self.causal_mask(T, tokens.device)

        # 4. Pass through Transformer blocks.
        for blk in self.blocks:
            x = blk(x, context=image_feats, mask=mask)

        # 5. Final normalization and projection to vocabulary.
        x = self.norm(x)
        logits = self.lm_head(x)
        logger.debug(f"TextDecoder output logits shape: {logits.shape}")

        return logits
