"""High‑level VAE model that binds encoder, decoder, Gumbel module and LLM prior."""
from __future__ import annotations

from typing import Dict
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Model
from config import VAEConfig
from .vision_encoder import VisionEncoder
from .text_decoder import TextDecoder
from .gumbel_softmax import GumbelSoftmax
from .diffusion.diffusion_decoder import DiffusionDecoder

class TextVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_enc = VisionEncoder(cfg)
        self.text_dec = TextDecoder(cfg)
        self.gumbel = GumbelSoftmax(cfg)
        self.llm = GPT2Model.from_pretrained(cfg.llm_model_name)
        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad_(False)
        self.diffusion = DiffusionDecoder(cfg, self.llm.wte)  # share embeddings
        self.register_buffer("start_token", torch.zeros(1, 1, cfg.text_decoder_dim))
        nn.init.trunc_normal_(self.start_token, std=0.02)

    def encode(self, img: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        img_feat = self.vision_enc(img)
        B = img.size(0)
        soft_tokens, hard_tokens, logits = [], [], []
        cur_emb = self.start_token.expand(B, -1, -1)
        for t in range(self.cfg.max_text_length):
            out = self.text_dec(cur_emb, img_feat)
            next_logit = out[:, -1:, :]
            logits.append(next_logit)
            if sample:
                soft = self.gumbel(next_logit, hard=True)
                soft_tokens.append(soft)
                hard_tokens.append(soft.argmax(-1))
                cur_emb = torch.cat([cur_emb, soft], dim=1)
        return {
            "logits": torch.cat(logits, 1),
            "soft_tokens": torch.cat(soft_tokens, 1) if sample else None,
            "hard_tokens": torch.cat(hard_tokens, 1) if sample else None,
        }

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.diffusion.sample(tokens)

    def compute_kl(self, logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        log_q = F.log_softmax(logits, -1)
        if tokens.dim() == 3:
            tokens = tokens.argmax(-1)
        with torch.no_grad():
            lm_out = self.llm(input_ids=tokens)
            lm_logits = lm_out.last_hidden_state @ self.llm.wte.weight.T
            log_p = F.log_softmax(lm_logits, -1)
        indices = tokens.unsqueeze(-1)
        log_q_sel = torch.gather(log_q, -1, indices).squeeze(-1)
        log_p_sel = torch.gather(log_p, -1, indices).squeeze(-1)
        return (log_q_sel - log_p_sel).sum(1).mean()

    def forward(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc = self.encode(img, sample=True)
        recon_loss = self.diffusion.p_losses(img, enc["soft_tokens"])
        kl = self.compute_kl(enc["logits"], enc["soft_tokens"])
        loss = recon_loss + self.cfg.kl_weight * kl
        self.gumbel.step()
        return {"loss": loss, "recon": recon_loss, "kl": kl}

def build_vae_from_config(cfg: VAEConfig) -> TextVAE:
    torch.manual_seed(cfg.seed)
    return TextVAE(cfg)
