# models/gumbel_softmax.py

"""Gumbel-Softmax sampling with temperature annealing."""
from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from torch import nn
from config import VAEConfig

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class GumbelSoftmax(nn.Module):
    """
    Implements the Gumbel-Softmax trick for sampling from a categorical
    distribution in a differentiable way. Includes temperature annealing.

    Args:
        cfg (VAEConfig): Configuration object containing Gumbel parameters.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        # The temperature controls the sharpness of the softmax. High temp -> uniform, low temp -> one-hot.
        self.register_buffer("temperature", torch.tensor(cfg.gumbel_temperature))
        logger.info(
            f"GumbelSoftmax initialized with temp: {self.temperature.item():.4f}, anneal rate: {cfg.gumbel_anneal_rate}"
        )

    @staticmethod
    def sample_gumbel(
        shape: torch.Size, device: torch.device, eps: float = 1e-20
    ) -> torch.Tensor:
        """
        Samples from the Gumbel distribution G(0, 1).

        This is achieved by transforming a uniform sample using the inverse CDF.
        The small epsilon `eps` is added for numerical stability to avoid log(0).
        """
        # Sample from Uniform(0, 1)
        u = torch.rand(shape, device=device)
        # Apply inverse transform sampling
        return -torch.log(-torch.log(u + eps) + eps)

    def forward(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Applies the Gumbel-Softmax trick.

        Args:
            logits (torch.Tensor): The raw output from a model layer. Shape: [..., vocab_size].
            hard (bool): If True, returns a one-hot vector on the forward pass,
                         but uses the soft, differentiable version for the backward pass.

        Returns:
            torch.Tensor: The sampled, differentiable output. Shape is same as logits.
        """
        # 1. Sample Gumbel noise and add it to the logits.
        gumbel_noise = self.sample_gumbel(logits.shape, logits.device)
        y = (
            logits + gumbel_noise
        ) / self.temperature.item()  # avoid in-place operation

        # 2. Apply softmax to get the soft, differentiable sample.
        y_soft = F.softmax(y, dim=-1)

        # 3. Use the straight-through estimator if `hard` is True.
        if hard or self.cfg.use_straight_through:
            # On the forward pass, we take the argmax to get a discrete one-hot vector.
            index = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(
                y_soft, memory_format=torch.contiguous_format
            ).scatter_(-1, index, 1.0)
            # On the backward pass, we detach the hard sample and add the soft sample.
            # This makes the gradient flow through y_soft, "tricking" the optimizer.
            y = y_hard - y_soft.detach() + y_soft
        else:
            y = y_soft

        return y

    def step(self) -> None:
        """
        Anneals the temperature for the next training step.
        This should be called once per training iteration (e.g., at the end of `forward`).
        """
        current_temp = self.temperature.item()
        # Decrease temperature by the anneal rate, but not below the minimum.
        new_temp = max(
            self.cfg.gumbel_temperature_min,
            current_temp
            * (1 - self.cfg.gumbel_anneal_rate),  # Using a multiplicative decay
        )
        self.temperature.fill_(new_temp)
        # Log only if the temperature has changed to avoid spamming logs
        if abs(new_temp - current_temp) > 1e-6:
            logger.debug(f"Gumbel temperature annealed to: {new_temp:.4f}")
