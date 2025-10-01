# models/diffusion/diffusion_decoder.py

"""DDPM wrapper that ties UNet with beta schedule and sampling utilities."""
from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from torch import nn
from config import VAEConfig
from .unet import UNet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DiffusionDecoder(nn.Module):
    """
    Manages the diffusion and reverse diffusion (sampling) processes.
    """

    def __init__(self, cfg: VAEConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.embed = embedding_layer
        self.net = UNet(cfg)

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.diffusion_timesteps)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_acum", torch.sqrt(alphas_cum))
        self.register_buffer("sqrt_one_minus_acum", torch.sqrt(1.0 - alphas_cum))

    def token_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 2:
            return self.embed(tokens)
        if tokens.dim() == 3:
            return tokens @ self.embed.weight
        raise ValueError(f"Unsupported token tensor dimensionality: {tokens.dim()}")

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha_cum_t = self.sqrt_acum[t, None, None, None]
        sqrt_one_minus_alpha_cum_t = self.sqrt_one_minus_acum[t, None, None, None]
        return sqrt_alpha_cum_t * x0 + sqrt_one_minus_alpha_cum_t * noise

    def p_losses(self, x0: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        t = torch.randint(
            0, self.cfg.diffusion_timesteps, (B,), device=x0.device, dtype=torch.long
        )
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        txt_cond = self.token_embed(tokens)
        predicted_noise = self.net(x_t, t, txt_cond)
        loss = F.mse_loss(predicted_noise, noise)
        logger.debug(f"Diffusion p_losses - Batch Size: {B}, Loss: {loss.item():.4f}")
        return loss

    @torch.no_grad()
    def sample(
        self, tokens: torch.Tensor, shape: tuple[int, ...] | None = None
    ) -> torch.Tensor:
        B = tokens.size(0)
        if shape is None:
            shape = (
                B,
                self.cfg.image_channels,
                self.cfg.image_size,
                self.cfg.image_size,
            )

        x = torch.randn(shape, device=tokens.device)
        txt_cond = self.token_embed(tokens)
        logger.info(
            f"Starting sampling for batch size {B} over {self.cfg.diffusion_timesteps} timesteps."
        )

        for t in reversed(range(self.cfg.diffusion_timesteps)):
            logger.debug(f"Sampling step t={t}")
            t_tensor = torch.full((B,), t, device=x.device, dtype=torch.long)
            pred_noise = self.net(x, t_tensor, txt_cond)

            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_acum_t = self.sqrt_one_minus_acum[t]

            term1 = 1 / sqrt_alpha_t
            term2 = beta_t / sqrt_one_minus_acum_t
            x = term1 * (x - term2 * pred_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        logger.info("Sampling complete.")
        return x.clamp(-1, 1)
