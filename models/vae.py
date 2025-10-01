# models/vae.py

"""High-level VAE model that binds encoder, decoder, Gumbel module and LLM prior."""
from __future__ import annotations
import logging
from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Model
from config import VAEConfig
from .vision_encoder import VisionEncoder
from .swin_encoder import SwinVisionEncoder
from .text_decoder import TextDecoder
from .gumbel_softmax import GumbelSoftmax
from .diffusion.diffusion_decoder import DiffusionDecoder

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAE(nn.Module):
    """
    The main Variational Autoencoder model.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # --- Initialize Components ---
        # **MODIFIED**: Select the vision encoder based on the config
        if cfg.vision_encoder_type == "vit":
            logger.info("Using Vision Transformer (ViT) as the vision encoder.")
            self.vision_enc = VisionEncoder(cfg)
        elif cfg.vision_encoder_type == "swin":
            logger.info("Using Swin Transformer V2 as the vision encoder.")
            self.vision_enc = SwinVisionEncoder(cfg)
        else:
            raise ValueError(
                f"Unknown vision_encoder_type: '{cfg.vision_encoder_type}'"
            )

        self.text_dec = TextDecoder(cfg)
        self.gumbel = GumbelSoftmax(cfg)

        try:
            logger.info(f"Loading LLM prior: {cfg.llm_model_name}")
            self.llm = GPT2Model.from_pretrained(cfg.llm_model_name)
        except (OSError, ValueError) as e:
            logger.error(
                f"Failed to load LLM '{cfg.llm_model_name}'. Check model name and internet connection."
            )
            raise e

        if cfg.freeze_llm:
            logger.info("Freezing LLM parameters.")
            for p in self.llm.parameters():
                p.requires_grad_(False)
            self.llm.eval()

        self.diffusion = DiffusionDecoder(cfg, self.llm.wte)
        self.register_buffer("start_token", torch.zeros(1, 1, cfg.text_decoder_dim))
        nn.init.trunc_normal_(self.start_token, std=0.02)

    def encode(
        self, img: torch.Tensor, sample: bool = True
    ) -> Dict[str, torch.Tensor | None]:
        img_feat = self.vision_enc(img)
        B = img.size(0)
        logits, soft_tokens, hard_tokens = [], [], []
        current_input_emb = self.start_token.expand(B, -1, -1)

        for t in range(self.cfg.max_text_length):
            output_logits = self.text_dec(current_input_emb, img_feat)[:, -1:, :]
            logits.append(output_logits)

            if not sample:
                continue

            next_soft_token = self.gumbel(output_logits, hard=True)
            soft_tokens.append(next_soft_token)
            hard_tokens.append(next_soft_token.argmax(-1))

            next_token_emb = next_soft_token @ self.text_dec.token_emb.weight
            current_input_emb = torch.cat([current_input_emb, next_token_emb], dim=1)

        results = {"logits": torch.cat(logits, dim=1)}
        if sample:
            results["soft_tokens"] = torch.cat(soft_tokens, dim=1)
            results["hard_tokens"] = torch.cat(hard_tokens, dim=1)
        else:
            results["soft_tokens"] = None
            results["hard_tokens"] = None
        return results

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        logger.info("Starting decoding (image generation) from tokens.")
        return self.diffusion.sample(tokens)

    def compute_kl(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 3:
            hard_tokens = tokens.argmax(-1)
        else:
            hard_tokens = tokens
        assert hard_tokens.dim() == 2, "Tokens must be 2D for KL computation"

        log_q = F.log_softmax(logits, dim=-1)
        log_q_sel = torch.gather(log_q, -1, hard_tokens.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            lm_out = self.llm(input_ids=hard_tokens)
            lm_logits = lm_out.last_hidden_state @ self.llm.wte.weight.T
            log_p = F.log_softmax(lm_logits, dim=-1)
            log_p_sel = torch.gather(log_p, -1, hard_tokens.unsqueeze(-1)).squeeze(-1)

        kl_div = (log_q_sel - log_p_sel).sum(dim=1).mean()
        return kl_div

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_out = self.encode(img, sample=True)
        soft_tokens = enc_out["soft_tokens"]
        assert soft_tokens is not None, "Encoding failed to produce soft tokens"

        recon_loss = self.diffusion.p_losses(img, soft_tokens)
        kl_loss = self.compute_kl(enc_out["logits"], soft_tokens)
        loss = recon_loss + self.cfg.kl_weight * kl_loss
        self.gumbel.step()

        # **MODIFIED**: Changed logger.info to logger.debug to reduce console spam
        logger.debug(
            f"Forward pass completed. Loss: {loss.item():.4f}, "
            f"Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}"
        )
        return {"loss": loss, "recon": recon_loss, "kl": kl_loss}


def build_vae_from_config(cfg: VAEConfig) -> TextVAE:
    logger.info(f"Building TextVAE model with seed: {cfg.seed}")
    torch.manual_seed(cfg.seed)
    return TextVAE(cfg)
