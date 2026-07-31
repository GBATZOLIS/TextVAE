"""§3 — Differentiable sampling with Gumbel-Softmax.

Implements the paper's §3.2 verbatim:

1. ``g = -log(-log u)``, ``u ~ Uniform(0,1)``           (Gumbel(0,1) inverse transform)
2. the Gumbel-max trick                                  (validated in ``tests/test_gumbel.py``)
3. Eq. 3: ``y_{t,k} = softmax_k((l_{t,k} + g_{t,k}) / τ)``
4. the optional straight-through estimator (hard forward, soft backward)

plus the τ annealing schedule of §3.2 ("anneal τ from e.g. 1.0 down to 0.1").
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..utils.config import GumbelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "sample_gumbel_noise",
    "gumbel_softmax_sample",
    "GumbelSoftmaxSample",
    "TemperatureScheduler",
]

# Guards ``log(u)`` against ``u == 0`` (which would produce ``inf`` noise).
_EPS = torch.finfo(torch.float32).tiny


@dataclass
class GumbelSoftmaxSample:
    """Result of one Gumbel-Softmax draw.

    Attributes
    ----------
    y:
        The relaxed (or straight-through hard) sample on the simplex, shape ``(..., V)``.
        This is the ``y_t`` of Eq. 3 and what gets embedded via ``E^T y`` (§5.1).
    y_soft:
        The purely soft Eq. 3 sample (always differentiable, never hardened).
    indices:
        ``argmax_k y_{.,k}`` — the discrete token ids of the latent sequence.
    log_probs:
        ``log softmax(logits)``, i.e. ``log q_φ(z_t = k | z_<t, x)`` of Eq. 2.
    """

    y: torch.Tensor
    y_soft: torch.Tensor
    indices: torch.Tensor
    log_probs: torch.Tensor


def sample_gumbel_noise(
    shape: torch.Size,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw ``Gumbel(0, 1)`` noise via ``g = -log(-log u)``  (§3.2 step 1).

    The noise is *not* part of the autograd graph: the whole point of the
    reparameterisation is that randomness is externalised (§3.2 step 2).
    """
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    u = u.clamp_min(_EPS)
    # ``-log(u) > 0``; clamp it (not ``log(u)``) before the outer log.
    return -torch.log((-torch.log(u)).clamp_min(_EPS))


def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    straight_through: bool = True,
    generator: Optional[torch.Generator] = None,
    noise: Optional[torch.Tensor] = None,
) -> GumbelSoftmaxSample:
    """Eq. 3 — a differentiable relaxed categorical sample over the vocabulary.

    Parameters
    ----------
    logits:
        Encoder logits ``l`` of shape ``(..., V)`` (unnormalised; Eq. 2).
    temperature:
        ``τ > 0``. As ``τ → 0`` the sample approaches a one-hot argmax draw; as
        ``τ → ∞`` it approaches the uniform distribution.
    hard:
        Return a discrete one-hot vector in the forward pass.
    straight_through:
        Only meaningful with ``hard=True``. When ``True`` the backward pass uses
        ``∇y_soft`` (§3.2 step 4); when ``False`` the hard sample is fully detached and no
        gradient reaches ``logits`` through ``y``.
    noise:
        Optional pre-drawn Gumbel noise (used by the tests to check determinism); when
        ``None`` fresh noise is drawn.
    """
    if temperature <= 0:
        raise ValueError(f"Gumbel-Softmax temperature must be > 0, got {temperature}")
    if logits.shape[-1] < 1:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    if noise is None:
        noise = sample_gumbel_noise(
            logits.shape, device=logits.device, dtype=logits.dtype, generator=generator
        )
    else:
        noise = noise.to(device=logits.device, dtype=logits.dtype)

    # Eq. 3.  Perturbed logits are divided by τ before the softmax.
    y_soft = torch.softmax((logits + noise) / temperature, dim=-1)
    indices = y_soft.argmax(dim=-1)

    if hard:
        y_hard = torch.zeros_like(y_soft).scatter_(-1, indices.unsqueeze(-1), 1.0)
        # Straight-through: identical value to ``y_hard``, gradient of ``y_soft``.
        y = (y_hard - y_soft.detach() + y_soft) if straight_through else y_hard
    else:
        y = y_soft

    # log q_φ(z_t = · | z_<t, x) — Eq. 2, computed from the *un-perturbed* logits.
    log_probs = torch.log_softmax(logits, dim=-1)
    return GumbelSoftmaxSample(y=y, y_soft=y_soft, indices=indices, log_probs=log_probs)


class TemperatureScheduler:
    """τ annealing (§3.2): ``constant``, ``linear``, ``exponential`` or ``cosine``.

    The schedule is a pure function of the global step, and its (trivial) state is
    checkpointed so that resumed runs continue with the same τ.
    """

    def __init__(
        self,
        schedule: str = "exponential",
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        anneal_steps: int = 50_000,
        warmup_steps: int = 0,
    ) -> None:
        if schedule not in {"constant", "linear", "exponential", "cosine"}:
            raise ValueError(f"unknown temperature schedule {schedule!r}")
        if tau_start <= 0 or tau_end <= 0:
            raise ValueError("temperatures must be > 0")
        if anneal_steps < 0 or warmup_steps < 0:
            raise ValueError("step counts must be >= 0")
        self.schedule = schedule
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_steps = int(anneal_steps)
        self.warmup_steps = int(warmup_steps)
        self.last_step = 0

    @classmethod
    def from_config(cls, cfg: GumbelConfig) -> "TemperatureScheduler":
        return cls(
            schedule=cfg.schedule,
            tau_start=cfg.tau_start,
            tau_end=cfg.tau_end,
            anneal_steps=cfg.anneal_steps,
            warmup_steps=cfg.warmup_steps,
        )

    def value(self, step: int) -> float:
        """τ at ``step`` (holds ``tau_start`` during warmup, ``tau_end`` afterwards)."""
        step = max(0, int(step))
        if self.schedule == "constant":
            return self.tau_start
        if step < self.warmup_steps:
            return self.tau_start
        if self.anneal_steps == 0:
            return self.tau_end
        progress = min(1.0, (step - self.warmup_steps) / self.anneal_steps)
        if self.schedule == "linear":
            return self.tau_start + progress * (self.tau_end - self.tau_start)
        if self.schedule == "exponential":
            # Geometric interpolation: τ(p) = τ_0 * (τ_1/τ_0)^p.
            return self.tau_start * (self.tau_end / self.tau_start) ** progress
        # cosine
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.tau_end + (self.tau_start - self.tau_end) * cosine

    def step(self, step: Optional[int] = None) -> float:
        """Advance (or jump to) ``step`` and return the new τ."""
        self.last_step = self.last_step + 1 if step is None else int(step)
        return self.value(self.last_step)

    @property
    def current(self) -> float:
        return self.value(self.last_step)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schedule": self.schedule,
            "tau_start": self.tau_start,
            "tau_end": self.tau_end,
            "anneal_steps": self.anneal_steps,
            "warmup_steps": self.warmup_steps,
            "last_step": self.last_step,
        }

    #: Fields describing the *shape* of the schedule, as opposed to its position.
    _HYPERPARAMETERS = (
        "schedule",
        "tau_start",
        "tau_end",
        "anneal_steps",
        "warmup_steps",
    )

    def load_state_dict(
        self, state: Dict[str, Any], restore_hyperparameters: bool = False
    ) -> None:
        """Restore the schedule's *position*; its shape comes from the current config.

        As with :class:`~project.training.losses.BetaScheduler`, resuming must not silently
        reinstate the checkpoint's τ schedule over one the user just reconfigured. Pass
        ``restore_hyperparameters=True`` to reproduce a checkpoint's schedule exactly.
        """
        if restore_hyperparameters:
            self.schedule = state.get("schedule", self.schedule)
            self.tau_start = float(state.get("tau_start", self.tau_start))
            self.tau_end = float(state.get("tau_end", self.tau_end))
            self.anneal_steps = int(state.get("anneal_steps", self.anneal_steps))
            self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        else:
            changed = {
                key: (state[key], getattr(self, key))
                for key in self._HYPERPARAMETERS
                if key in state and state[key] != getattr(self, key)
            }
            if changed:
                logger.info(
                    "Temperature schedule differs from the checkpoint; keeping the configured "
                    "values (checkpoint -> config: %s).",
                    ", ".join(
                        f"{k}: {old} -> {new}" for k, (old, new) in changed.items()
                    ),
                )
        self.last_step = int(state.get("last_step", self.last_step))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TemperatureScheduler(schedule={self.schedule!r}, tau_start={self.tau_start}, "
            f"tau_end={self.tau_end}, anneal_steps={self.anneal_steps}, "
            f"warmup_steps={self.warmup_steps}, last_step={self.last_step})"
        )
