"""Seeding and RNG-state capture (needed for exact training resumption)."""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["set_seed", "get_rng_state", "set_rng_state"]


def set_seed(seed: int, rank: int = 0, deterministic: bool = False) -> None:
    """Seed python/numpy/torch. ``rank`` offsets the seed so workers differ."""
    effective = seed + rank
    os.environ["PYTHONHASHSEED"] = str(effective)
    random.seed(effective)
    np.random.seed(effective % (2**32))
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> Dict[str, Any]:
    """Snapshot every RNG we touch, for checkpointing."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    """Restore a snapshot produced by :func:`get_rng_state`."""
    if not state:
        return
    if "python" in state:
        random.setstate(_as_python_state(state["python"]))
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_as_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = [_as_byte_tensor(s) for s in state["cuda"]]
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:  # different GPU count than the run we resume from
            logger.warning(
                "Skipping CUDA RNG restore: checkpoint has %d device states, found %d devices.",
                len(cuda_states),
                torch.cuda.device_count(),
            )


def _as_python_state(state: Any) -> Any:
    # ``torch.save`` round-trips tuples faithfully, but be forgiving about lists.
    if isinstance(state, list):
        return (state[0], tuple(state[1]), state[2])
    return state


def _as_byte_tensor(state: Any) -> torch.Tensor:
    tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state)
    return tensor.to(dtype=torch.uint8, device="cpu")
