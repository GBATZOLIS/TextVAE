# models/vae.py

"""
Main VAE model with a Swin Transformer vision encoder.
This version includes a robust method for determining the vision
encoder's output dimension to prevent dimension mismatch errors.
"""
from __future__ import annotations
import logging
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, SwinModel

from config import VAEConfig
from .text_decoder import TextDecoder
from .gumbel_softmax import GumbelSoftmax
from .diffusion.diffusion_decoder import DiffusionDecoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAE(nn.Module):
    """The End-to-End Text-Conditioned VAE with a Swin Transformer."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        try:
            # --- 1. Initialize Vision Encoder ---
            logger.info(
                f"Loading Vision Encoder to fine-tune: {cfg.vision_encoder_model}"
            )
            self.vision_enc_model = SwinModel.from_pretrained(cfg.vision_encoder_model)

            # --- 2. Initialize Linguistic Components ---
            logger.info(f"Loading LLM prior (for KL): {cfg.llm_prior_model}")
            self.llm_prior = GPT2Model.from_pretrained(cfg.llm_prior_model)
            self.tokenizer = GPT2Tokenizer.from_pretrained(cfg.llm_prior_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except OSError as e:
            logger.error(
                f"Could not load a pretrained model. Please check model names and internet connection. Error: {e}"
            )
            raise

        # --- 3. Initialize Trainable Components ---
        self.gumbel = GumbelSoftmax(cfg)
        self.text_dec = TextDecoder(
            cfg,
            token_embedding_layer=self.llm_prior.wte,
            vocab_size=self.llm_prior.config.vocab_size,
        )

        # FIX: Robustly determine the Swin encoder's output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, cfg.image_channels, cfg.image_size, cfg.image_size
            )
            dummy_output = self.vision_enc_model(dummy_input).last_hidden_state
            swin_output_dim = dummy_output.shape[-1]
        logger.info(f"Robustly determined Swin output dimension: {swin_output_dim}")

        # Projection layer to bridge vision and text dimensions
        self.vision_feat_proj = nn.Linear(swin_output_dim, cfg.text_decoder_dim)

        # --- 4. Initialize Trainable Diffusion Decoder ---
        logger.info("Initializing custom trainable Diffusion Decoder.")
        self.diffusion_dec = DiffusionDecoder(cfg, embedding_layer=self.llm_prior.wte)

        # --- 5. Set Operational Flags ---
        self._set_requires_grad()
        self.start_token_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else 50256
        )

    def _set_requires_grad(self):
        """Helper to set requires_grad flags based on config."""
        if self.cfg.freeze_vision_encoder:
            self.vision_enc_model.requires_grad_(False)
            logger.info("Froze Vision Encoder parameters.")
        if self.cfg.freeze_llm_prior:
            self.llm_prior.requires_grad_(False)
            logger.info("Froze LLM prior (GPT-2) parameters.")
        if self.cfg.freeze_diffusion_decoder:
            self.diffusion_dec.requires_grad_(False)
            logger.info("Froze custom Diffusion Decoder parameters.")

    def compute_kl(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Computes KL divergence against the frozen GPT-2 prior."""
        log_q = F.log_softmax(logits, dim=-1)
        pad_mask = (tokens != self.tokenizer.pad_token_id).float()
        log_q_sel = torch.gather(log_q, -1, tokens.unsqueeze(-1)).squeeze(-1) * pad_mask

        with torch.no_grad():
            lm_out = self.llm_prior(input_ids=tokens).last_hidden_state
            lm_logits = self.text_dec.lm_head(lm_out)
            log_p = F.log_softmax(lm_logits, dim=-1)
            log_p_sel = (
                torch.gather(log_p, -1, tokens.unsqueeze(-1)).squeeze(-1) * pad_mask
            )

        kl_div = (log_q_sel - log_p_sel).sum(dim=1)
        return kl_div.mean()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """The main forward pass for training the end-to-end model."""
        img_feat_raw = self.vision_enc_model(img).last_hidden_state
        img_feat = self.vision_feat_proj(img_feat_raw)

        B = img.size(0)
        hard_tokens = torch.full(
            (B, 1), self.start_token_id, dtype=torch.long, device=img.device
        )
        soft_tokens_dist = []
        all_logits = []

        for _ in range(self.cfg.max_text_length - 1):
            logits, _ = self.text_dec(hard_tokens, img_feat)
            next_token_logits = logits[:, -1, :]

            soft_next_token = self.gumbel(
                next_token_logits, hard=self.cfg.use_straight_through
            )
            hard_next_token = soft_next_token.argmax(dim=-1, keepdim=True)

            hard_tokens = torch.cat([hard_tokens, hard_next_token], dim=1)
            soft_tokens_dist.append(soft_next_token.unsqueeze(1))
            all_logits.append(next_token_logits.unsqueeze(1))

        logits = torch.cat(all_logits, dim=1)
        soft_tokens_dist = torch.cat(soft_tokens_dist, dim=1)

        recon_loss = self.diffusion_dec.p_losses(img, tokens=soft_tokens_dist)
        kl_loss = self.compute_kl(logits, hard_tokens[:, 1:])

        loss = recon_loss + self.cfg.kl_weight * kl_loss
        self.gumbel.step()

        return {"loss": loss, "recon": recon_loss, "kl": kl_loss}


def build_vae_from_config(cfg: VAEConfig) -> TextVAE:
    """Factory function to build the VAE model from a config object."""
    logger.info(f"Building TextVAE model with seed: {cfg.seed}")
    torch.manual_seed(cfg.seed)
    # Potentially add CUDA seed if using GPU
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(cfg.seed)
    return TextVAE(cfg)
