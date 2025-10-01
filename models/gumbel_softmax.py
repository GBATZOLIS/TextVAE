# models/gumbel_softmax.py

"""Gumbel-Softmax sampling with temperature annealing."""
from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from torch import nn
from config import VAEConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class GumbelSoftmax(nn.Module):
    """
    Implements the Gumbel-Softmax trick for sampling from a categorical
    distribution in a differentiable way. Includes temperature annealing.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("temperature", torch.tensor(cfg.gumbel_temperature))
        logger.info(
            f"GumbelSoftmax initialized with temp: {self.temperature.item():.4f}, anneal rate: {cfg.gumbel_anneal_rate}"
        )

    @staticmethod
    def sample_gumbel(
        shape: torch.Size, device: torch.device, eps: float = 1e-20
    ) -> torch.Tensor:
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + eps) + eps)

    def forward(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        gumbel_noise = self.sample_gumbel(logits.shape, logits.device)
        y = (logits + gumbel_noise) / self.temperature
        y_soft = F.softmax(y, dim=-1)

        if hard or self.cfg.use_straight_through:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(
                y_soft, memory_format=torch.contiguous_format
            ).scatter_(-1, index, 1.0)
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        return y

    def step(self) -> None:
        """Anneals the temperature for the next training step."""
        current_temp = self.temperature.item()
        new_temp = max(
            self.cfg.gumbel_temperature_min,
            current_temp * (1 - self.cfg.gumbel_anneal_rate),
        )
        self.temperature.fill_(new_temp)
        if abs(new_temp - current_temp) > 1e-6:
            logger.debug(f"Gumbel temperature annealed to: {new_temp:.4f}")
