"""DDPM wrapper that ties UNet with beta schedule and sampling utilities."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from config import VAEConfig
from .unet import UNet

class DiffusionDecoder(nn.Module):
    def __init__(self, cfg: VAEConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.embed = embedding_layer  # shared with LLM
        self.net = UNet(cfg)
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_acum", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_one_minus_acum", torch.sqrt(1 - alphas_cum))

    def token_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            return self.embed(tokens)
        return tokens @ self.embed.weight

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_acum[t, None, None, None] * x0 + self.sqrt_one_minus_acum[t, None, None, None] * noise

    def p_losses(self, x0: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(0, self.cfg.diffusion_timesteps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        txt = self.token_embed(tokens)
        pred = self.net(x_t, t, txt)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, tokens: torch.Tensor, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        B = tokens.size(0)
        if shape is None:
            shape = (B, self.cfg.image_channels, self.cfg.image_size, self.cfg.image_size)
        x = torch.randn(shape, device=tokens.device)
        txt = self.token_embed(tokens)
        for t in reversed(range(self.cfg.diffusion_timesteps)):
            noise = torch.randn_like(x) if t > 0 else 0
            pred_noise = self.net(x, torch.full((B,), t, device=x.device), txt)
            beta_t = self.betas[t]
            alpha_t = 1 - beta_t
            alpha_cum_t = self.sqrt_acum[t] ** 2
            x = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_cum_t) * pred_noise) + torch.sqrt(beta_t) * noise
        return x.clamp(-1, 1)
