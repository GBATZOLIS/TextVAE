"""Console logging plus a unified TensorBoard / Weights & Biases metric logger."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

__all__ = ["setup_logging", "MetricLogger"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    rank: int = 0,
) -> logging.Logger:
    """Configure the root logger. Non-zero ranks are demoted to WARNING."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    effective = level if rank == 0 else logging.WARNING
    formatter = logging.Formatter(_LOG_FORMAT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if log_file is not None and rank == 0:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.setLevel(effective)
    # Silence the noisiest third-party loggers.
    for noisy in ("PIL", "urllib3", "filelock", "fsspec", "deepspeed", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("textvae")


class MetricLogger:
    """Writes scalars/images/text to TensorBoard and/or W&B from the main process only.

    Both backends are optional: if the package is missing (or disabled in the config) the
    corresponding calls become no-ops, so training never depends on a tracker.
    """

    def __init__(
        self,
        log_dir: Union[str, Path],
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "text-latent-vae",
        wandb_entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        is_main_process: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.is_main_process = is_main_process
        self.logger = logging.getLogger("textvae.metrics")
        self._tb = None
        self._wandb = None

        if not is_main_process:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.log_dir))
            except Exception as exc:  # pragma: no cover - optional dependency
                self.logger.warning(
                    "TensorBoard unavailable (%s); continuing without it.", exc
                )
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb
                if wandb.run is None:
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=run_name,
                        dir=str(self.log_dir),
                        config=config,
                    )
            except Exception as exc:  # pragma: no cover - optional dependency
                self.logger.warning("W&B unavailable (%s); continuing without it.", exc)
                self._wandb = None

    # -- scalars ---------------------------------------------------------------------
    def log_scalars(
        self, metrics: Dict[str, float], step: int, prefix: str = ""
    ) -> None:
        if not self.is_main_process or not metrics:
            return
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items() if v is not None}
        if self._tb is not None:
            for key, value in flat.items():
                self._tb.add_scalar(key, value, step)
        if self._wandb is not None:
            self._wandb.log(flat, step=step)

    def log_console(self, metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
        if not self.is_main_process:
            return
        parts = []
        for key, value in metrics.items():
            parts.append(
                f"{key}={value:.4g}"
                if isinstance(value, (int, float))
                else f"{key}={value}"
            )
        self.logger.info("%sstep %d | %s", prefix, step, " ".join(parts))

    # -- images / text ---------------------------------------------------------------
    def log_images(self, tag: str, images: torch.Tensor, step: int) -> None:
        """``images``: float tensor in [0, 1], shape (B, C, H, W)."""
        if not self.is_main_process or images.numel() == 0:
            return
        images = images.detach().float().clamp(0, 1).cpu()
        if self._tb is not None:
            self._tb.add_images(tag, images, step)
        if self._wandb is not None:
            self._wandb.log(
                {tag: [self._wandb.Image(img) for img in images]}, step=step
            )

    def log_text(self, tag: str, lines: Sequence[str], step: int) -> None:
        if not self.is_main_process or not lines:
            return
        body = "\n\n".join(f"{i}. {line}" for i, line in enumerate(lines))
        if self._tb is not None:
            self._tb.add_text(tag, body, step)
        if self._wandb is not None:
            table = self._wandb.Table(columns=["index", "latent_text"])
            for i, line in enumerate(lines):
                table.add_data(i, line)
            self._wandb.log({tag: table}, step=step)

    def log_reconstruction_panel(
        self,
        tag: str,
        originals: torch.Tensor,
        reconstructions: torch.Tensor,
        latent_texts: List[str],
        step: int,
    ) -> None:
        """Log originals, reconstructions and the latent sentences that produced them."""
        self.log_images(f"{tag}/original", originals, step)
        self.log_images(f"{tag}/reconstruction", reconstructions, step)
        self.log_text(f"{tag}/latent_text", latent_texts, step)

    # -- lifecycle -------------------------------------------------------------------
    def flush(self) -> None:
        if self._tb is not None:
            self._tb.flush()

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None
        if self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()
