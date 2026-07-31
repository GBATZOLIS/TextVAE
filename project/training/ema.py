"""Exponential moving average of the trainable parameters (optional, off by default)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["ExponentialMovingAverage"]


class ExponentialMovingAverage:
    """Tracks ``ema ← decay·ema + (1-decay)·param`` for every trainable parameter.

    The shadow weights usually reconstruct noticeably better than the raw ones, so
    :class:`~project.training.trainer.Trainer` can evaluate and checkpoint with them
    (``train.ema.use_for_eval``).
    """

    def __init__(
        self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone().float()
            for name, param in self._trainable(model)
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _trainable(model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                yield name, param

    def current_decay(self) -> float:
        """Ramp the decay in during warmup so early averages are not dominated by init."""
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.current_decay()
        for name, param in self._trainable(model):
            value = param.detach().float()
            shadow = self.shadow.get(name)
            if shadow is None or shadow.shape != value.shape:
                self.shadow[name] = value.clone()
                continue
            if shadow.device != value.device:
                # Happens when the state was restored from a CPU checkpoint, or when the
                # model was moved after the EMA was created.
                shadow = self.shadow[name] = shadow.to(value.device)
            shadow.mul_(decay).add_(value, alpha=1.0 - decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's trainable weights with the EMA weights (after :meth:`store`)."""
        for name, param in self._trainable(model):
            if name in self.shadow:
                param.copy_(
                    self.shadow[name].to(dtype=param.dtype, device=param.device)
                )

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Snapshot the live weights so :meth:`restore` can undo :meth:`copy_to`."""
        self._backup = {
            name: param.detach().clone() for name, param in self._trainable(model)
        }

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if not self._backup:
            return
        for name, param in self._trainable(model):
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.num_updates = int(state.get("num_updates", 0))
        shadow = state.get("shadow", {})
        missing = set(self.shadow) - set(shadow)
        if missing:
            logger.warning(
                "EMA checkpoint is missing %d parameter(s); keeping current values.",
                len(missing),
            )
        for name, tensor in shadow.items():
            self.shadow[name] = tensor.detach().clone().float()

    class _Swapped:
        """Context manager applying the EMA weights temporarily."""

        def __init__(self, ema: "ExponentialMovingAverage", model: nn.Module) -> None:
            self.ema, self.model = ema, model

        def __enter__(self) -> nn.Module:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
            return self.model

        def __exit__(self, *exc: Any) -> None:
            self.ema.restore(self.model)

    def applied_to(self, model: nn.Module) -> "ExponentialMovingAverage._Swapped":
        """``with ema.applied_to(model): ...`` — evaluate with the shadow weights."""
        return self._Swapped(self, model)
