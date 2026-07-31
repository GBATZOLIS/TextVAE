"""§4.2 + Eq. 8 — the KL Monte-Carlo estimator, β scheduling and ELBO assembly.

The KL term is computed exactly as the paper prescribes: for a *single* sampled latent
sequence,

    KL(q_φ(z|x) ‖ p(z)) ≈ log q_φ(z | x) − log p(z),

with ``log q`` from the encoder logits (Eq. 2) and ``log p`` from the frozen LM (§4.1).
No surrogate, no closed form, no substitution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from ..utils.config import LossConfig

logger = logging.getLogger(__name__)

__all__ = [
    "kl_divergence_mc",
    "KLOutput",
    "BetaScheduler",
    "ELBOTerms",
    "assemble_elbo",
]


@dataclass
class KLOutput:
    """Per-sample and reduced KL values."""

    per_sample: (
        torch.Tensor
    )  # (B,) the (possibly rescaled / floored) KL used in the loss
    raw_per_sample: torch.Tensor  # (B,) log q - log p, exactly as in §4.2
    mean: torch.Tensor  # scalar, mean over the batch (§6 step 4)


def kl_divergence_mc(
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    reduction: str = "sum",
    latent_length: Optional[int] = None,
    free_bits: float = 0.0,
) -> KLOutput:
    """One-sample MC estimate of ``KL(q ‖ p)`` (Eqs. 4–6).

    Parameters
    ----------
    log_q, log_p:
        ``(B,)`` sequence-level log-probabilities, each already summed over ``t``.
    reduction:
        ``sum`` keeps Eq. 6's sum over positions. ``mean_per_token`` divides by ``T``,
        which only rescales the term relative to the per-pixel diffusion MSE (a practical
        knob; ``β`` can express the same thing).
    latent_length:
        ``T``; required for ``mean_per_token`` and for ``free_bits``.
    free_bits:
        Optional floor in nats *per token* (``0.0`` disables it, giving exactly Eq. 8).
    """
    if log_q.shape != log_p.shape:
        raise ValueError(
            f"log_q shape {tuple(log_q.shape)} != log_p shape {tuple(log_p.shape)}"
        )
    if reduction not in {"sum", "mean_per_token"}:
        raise ValueError(f"unknown KL reduction {reduction!r}")

    raw = log_q - log_p.to(log_q.dtype)  # §4.2: the KL estimate for this sample
    per_sample = raw

    if reduction == "mean_per_token":
        if not latent_length:
            raise ValueError("latent_length is required for reduction='mean_per_token'")
        per_sample = per_sample / float(latent_length)

    if free_bits > 0.0:
        if not latent_length:
            raise ValueError("latent_length is required for free_bits > 0")
        floor = free_bits * (
            1.0 if reduction == "mean_per_token" else float(latent_length)
        )
        # Only penalise the amount by which the KL exceeds the floor.
        per_sample = torch.clamp(per_sample, min=floor)

    return KLOutput(per_sample=per_sample, raw_per_sample=raw, mean=per_sample.mean())


class BetaScheduler:
    """β schedule for Eq. 8.

    ``constant``
        β = ``beta`` at every step (the faithful ELBO when ``beta=1``).
    ``linear`` / ``cosine``
        Warm β up from ``beta_start`` to ``beta_end`` over ``beta_anneal_steps`` after
        ``beta_warmup_steps`` — the standard remedy for early posterior collapse.
    ``cyclical``
        Repeated linear warm-ups of length ``beta_cycle_steps``.
    """

    def __init__(
        self,
        schedule: str = "constant",
        beta: float = 1.0,
        beta_start: float = 0.0,
        beta_end: float = 1.0,
        warmup_steps: int = 0,
        anneal_steps: int = 10_000,
        cycle_steps: int = 10_000,
    ) -> None:
        if schedule not in {"constant", "linear", "cosine", "cyclical"}:
            raise ValueError(f"unknown beta schedule {schedule!r}")
        self.schedule = schedule
        self.beta = float(beta)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.warmup_steps = int(warmup_steps)
        self.anneal_steps = int(anneal_steps)
        self.cycle_steps = int(cycle_steps)
        self.last_step = 0

    @classmethod
    def from_config(cls, cfg: LossConfig) -> "BetaScheduler":
        return cls(
            schedule=cfg.beta_schedule,
            beta=cfg.beta,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            warmup_steps=cfg.beta_warmup_steps,
            anneal_steps=cfg.beta_anneal_steps,
            cycle_steps=cfg.beta_cycle_steps,
        )

    def value(self, step: int) -> float:
        step = max(0, int(step))
        if self.schedule == "constant":
            return self.beta
        if self.schedule == "cyclical":
            if self.cycle_steps <= 0:
                return self.beta_end
            progress = (step % self.cycle_steps) / self.cycle_steps
            return self.beta_start + progress * (self.beta_end - self.beta_start)
        if step < self.warmup_steps:
            return self.beta_start
        if self.anneal_steps <= 0:
            return self.beta_end
        progress = min(1.0, (step - self.warmup_steps) / self.anneal_steps)
        if self.schedule == "linear":
            return self.beta_start + progress * (self.beta_end - self.beta_start)
        # cosine: slow start, slow finish
        ramp = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.beta_start + ramp * (self.beta_end - self.beta_start)

    def step(self, step: Optional[int] = None) -> float:
        self.last_step = self.last_step + 1 if step is None else int(step)
        return self.value(self.last_step)

    @property
    def current(self) -> float:
        return self.value(self.last_step)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schedule": self.schedule,
            "beta": self.beta,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "warmup_steps": self.warmup_steps,
            "anneal_steps": self.anneal_steps,
            "cycle_steps": self.cycle_steps,
            "last_step": self.last_step,
        }

    #: Fields describing the *shape* of the schedule, as opposed to its position.
    _HYPERPARAMETERS = (
        "schedule",
        "beta",
        "beta_start",
        "beta_end",
        "warmup_steps",
        "anneal_steps",
        "cycle_steps",
    )

    def load_state_dict(
        self, state: Dict[str, Any], restore_hyperparameters: bool = False
    ) -> None:
        """Restore the schedule's *position*; its shape comes from the current config.

        A checkpoint stores both where the run had got to and how β was configured at the
        time. On resume the config must win: otherwise ``--set loss.beta=0.01 --resume ...``
        would silently keep the checkpoint's old β, and the run would carry on with the
        very setting the user was trying to change. Pass ``restore_hyperparameters=True``
        to reproduce a checkpoint's schedule exactly.
        """
        if restore_hyperparameters:
            for key in self._HYPERPARAMETERS:
                if key in state:
                    setattr(self, key, state[key])
        else:
            changed = {
                key: (state[key], getattr(self, key))
                for key in self._HYPERPARAMETERS
                if key in state and state[key] != getattr(self, key)
            }
            if changed:
                logger.info(
                    "Beta schedule differs from the checkpoint; keeping the configured values "
                    "(checkpoint -> config: %s).",
                    ", ".join(
                        f"{k}: {old} -> {new}" for k, (old, new) in changed.items()
                    ),
                )
        self.last_step = int(state.get("last_step", self.last_step))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BetaScheduler(schedule={self.schedule!r}, current={self.current:.4g})"


@dataclass
class ELBOTerms:
    """The scalar loss of Eq. 8 plus everything worth logging."""

    loss: torch.Tensor
    diffusion_loss: torch.Tensor
    kl: torch.Tensor
    beta: float
    extra: Dict[str, torch.Tensor] = field(default_factory=dict)


def assemble_elbo(
    diffusion_loss: torch.Tensor,
    kl: torch.Tensor,
    beta: float,
    extra_losses: Optional[Dict[str, torch.Tensor]] = None,
) -> ELBOTerms:
    """Eq. 8: ``L = mean(L_diff) + β · mean(L_KL)`` (+ optional auxiliary terms).

    Maximising the ELBO of Eq. 1 is minimising this quantity; the diffusion MSE stands in
    for ``-log p_θ(x | z)`` per §5.2.
    """
    total = diffusion_loss + beta * kl
    extras = extra_losses or {}
    for value in extras.values():
        total = total + value
    return ELBOTerms(
        loss=total,
        diffusion_loss=diffusion_loss,
        kl=kl,
        beta=float(beta),
        extra=dict(extras),
    )
