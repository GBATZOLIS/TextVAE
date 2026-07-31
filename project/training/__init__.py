"""Training stack: losses, schedules, EMA, checkpointing and the training loop.

``Trainer`` is imported lazily: ``project.models.vae`` depends on
``project.training.losses``, so eagerly importing the trainer here (which imports the
models) would create an import cycle.
"""

from typing import TYPE_CHECKING, Any

from .checkpoint import CheckpointManager, load_checkpoint, save_checkpoint
from .ema import ExponentialMovingAverage
from .losses import BetaScheduler, ELBOTerms, KLOutput, assemble_elbo, kl_divergence_mc

if TYPE_CHECKING:  # pragma: no cover
    from .trainer import Trainer

__all__ = [
    "CheckpointManager",
    "load_checkpoint",
    "save_checkpoint",
    "ExponentialMovingAverage",
    "BetaScheduler",
    "ELBOTerms",
    "KLOutput",
    "assemble_elbo",
    "kl_divergence_mc",
    "Trainer",
]


def __getattr__(name: str) -> Any:
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
