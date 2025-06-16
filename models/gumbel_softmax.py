"""Gumbel‑Softmax sampling with temperature annealing."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from config import VAEConfig

class GumbelSoftmax(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("temperature", torch.tensor(cfg.gumbel_temperature))

    @staticmethod
    def sample_gumbel(shape: torch.Size, device: torch.device) -> torch.Tensor:
        u = torch.rand(shape, device=device)
        return -torch.log(-torch.log(u + 1e-20) + 1e-20)

    def forward(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        gumbel_noise = self.sample_gumbel(logits.shape, logits.device)
        y = (logits + gumbel_noise) / self.temperature
        y_soft = F.softmax(y, dim=-1)
        if hard or self.cfg.use_straight_through:
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(2, index, 1.0)
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft
        return y

    def step(self) -> None:
        new_temp = max(
            self.cfg.gumbel_temperature_min,
            self.temperature.item() * (1 - self.cfg.gumbel_anneal_rate),
        )
        self.temperature.fill_(new_temp)
