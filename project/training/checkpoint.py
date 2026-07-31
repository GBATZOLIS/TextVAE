"""Checkpointing: save and restore the *entire* training state for exact resumption.

Saved: encoder + diffusion decoder weights (``φ`` and ``θ``), optimizer, LR scheduler,
EMA shadow weights, the τ and β schedules, global step / epoch, RNG states and the config.

Deliberately *not* saved: the frozen LM prior, a frozen vision backbone and the frozen
image VAE. Those modules return an empty ``state_dict()`` and are rebuilt from the config,
which keeps checkpoints small without losing any state (nothing about them ever changes).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..utils.config import Config, config_from_dict, config_to_dict
from ..utils.seed import get_rng_state, set_rng_state

logger = logging.getLogger(__name__)

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointManager",
    "LoadedCheckpoint",
]

_STEP_PATTERN = re.compile(r"step_(\d+)")


@dataclass
class LoadedCheckpoint:
    """Metadata recovered from a checkpoint."""

    global_step: int
    epoch: int
    config: Optional[Config]
    extra: Dict[str, Any]
    path: Path


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    temperature_scheduler: Optional[Any] = None,
    beta_scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    global_step: int = 0,
    epoch: int = 0,
    config: Optional[Config] = None,
    save_rng_state: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a full training-state checkpoint to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "format_version": 1,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    if temperature_scheduler is not None:
        payload["temperature_scheduler"] = temperature_scheduler.state_dict()
    if beta_scheduler is not None:
        payload["beta_scheduler"] = beta_scheduler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if config is not None:
        payload["config"] = config_to_dict(config)
    if save_rng_state:
        payload["rng_state"] = get_rng_state()
    if extra:
        payload["extra"] = extra

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-write never corrupts the checkpoint
    logger.info("Saved checkpoint %s (step %d).", path, global_step)
    return path


def load_checkpoint(
    path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    temperature_scheduler: Optional[Any] = None,
    beta_scheduler: Optional[Any] = None,
    ema: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
    restore_rng_state: bool = True,
) -> LoadedCheckpoint:
    """Restore everything that is passed in; returns the checkpoint's metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
        if missing:
            logger.warning(
                "Checkpoint missing %d key(s), e.g. %s", len(missing), missing[:3]
            )
        if unexpected:
            logger.warning(
                "Checkpoint has %d unexpected key(s), e.g. %s",
                len(unexpected),
                unexpected[:3],
            )
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in payload:
        lr_scheduler.load_state_dict(payload["lr_scheduler"])
    if temperature_scheduler is not None and "temperature_scheduler" in payload:
        temperature_scheduler.load_state_dict(payload["temperature_scheduler"])
    if beta_scheduler is not None and "beta_scheduler" in payload:
        beta_scheduler.load_state_dict(payload["beta_scheduler"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict(payload["ema"])
    if restore_rng_state and "rng_state" in payload:
        set_rng_state(payload["rng_state"])

    config: Optional[Config] = None
    if "config" in payload:
        try:
            config = config_from_dict(Config, payload["config"])
        except Exception as exc:  # pragma: no cover - config schema drift
            logger.warning("Could not rebuild the config stored in %s: %s", path, exc)

    logger.info("Loaded checkpoint %s (step %d).", path, payload.get("global_step", 0))
    return LoadedCheckpoint(
        global_step=int(payload.get("global_step", 0)),
        epoch=int(payload.get("epoch", 0)),
        config=config,
        extra=dict(payload.get("extra", {})),
        path=path,
    )


class CheckpointManager:
    """Writes ``step_<N>.pt`` files, prunes old ones and tracks the best metric."""

    def __init__(
        self,
        directory: Union[str, Path],
        keep_last_n: int = 3,
        best_metric_mode: str = "min",
    ) -> None:
        if best_metric_mode not in {"min", "max"}:
            raise ValueError("best_metric_mode must be 'min' or 'max'")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = int(keep_last_n)
        self.best_metric_mode = best_metric_mode
        self.best_metric: Optional[float] = None

    # ------------------------------------------------------------------- discovery
    def checkpoints(self) -> List[Path]:
        """All step checkpoints, oldest first."""
        numbered: List[Tuple[int, Path]] = []
        for path in self.directory.glob("step_*.pt"):
            match = _STEP_PATTERN.search(path.name)
            if match is not None:
                numbered.append((int(match.group(1)), path))
        return [path for _, path in sorted(numbered)]

    def latest(self) -> Optional[Path]:
        checkpoints = self.checkpoints()
        return checkpoints[-1] if checkpoints else None

    def best(self) -> Optional[Path]:
        path = self.directory / "best.pt"
        return path if path.exists() else None

    # ------------------------------------------------------------------------ save
    def save(
        self, global_step: int, metric: Optional[float] = None, **kwargs: Any
    ) -> Path:
        path = save_checkpoint(
            self.directory / f"step_{global_step}.pt", global_step=global_step, **kwargs
        )
        self._prune()
        if metric is not None and self._is_best(metric):
            self.best_metric = float(metric)
            best_path = self.directory / "best.pt"
            best_path.write_bytes(path.read_bytes())
            logger.info("New best checkpoint (metric=%.5f) -> %s", metric, best_path)
        return path

    def _is_best(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.best_metric_mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def _prune(self) -> None:
        if self.keep_last_n <= 0:
            return
        stale = self.checkpoints()[: -self.keep_last_n]
        for path in stale:
            try:
                path.unlink()
                logger.debug("Pruned old checkpoint %s", path)
            except OSError as exc:  # pragma: no cover - filesystem race
                logger.warning("Could not remove %s: %s", path, exc)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "best_metric_mode": self.best_metric_mode,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best_metric = state.get("best_metric")
        self.best_metric_mode = state.get("best_metric_mode", self.best_metric_mode)


def resolve_resume_path(resume: str, directory: Union[str, Path]) -> Optional[Path]:
    """Interpret ``train.resume``: ``""`` (fresh), ``auto`` (latest in ``directory``) or a path."""
    if not resume:
        return None
    if resume == "auto":
        return CheckpointManager(directory).latest()
    path = Path(resume)
    if path.is_dir():
        return CheckpointManager(path).latest()
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path
