"""Small distributed helpers.

Process-group setup itself is delegated to :mod:`accelerate` (see
:class:`project.training.trainer.Trainer`); these helpers only answer questions that the
model/dataset code needs without holding an ``Accelerator`` reference.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist

__all__ = [
    "is_distributed",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "is_main_process",
    "barrier",
    "all_reduce_mean",
]


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(
    value: torch.Tensor, group: Optional[object] = None
) -> torch.Tensor:
    """Average a tensor across ranks (no-op when running single-process)."""
    if not is_distributed():
        return value
    tensor = value.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor / get_world_size()
