"""The training loop of §6, wrapped in :mod:`accelerate` for AMP and multi-GPU.

Per optimisation step:

    batch ─► q_φ rollout (Gumbel, τ from the schedule) ─► Z' = E_LM^T y
          ─► L_diff (Eq. 7) ─► L_KL (Eq. 6) ─► L = L_diff + β·L_KL (Eq. 8)
          ─► backward ─► clip ─► optimizer ─► LR/τ/β schedules ─► EMA

with logging, periodic validation, qualitative sampling and full-state checkpointing.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..models.vae import TextVAE, TextVAEOutput
from ..utils.config import Config
from ..utils.logging_utils import MetricLogger
from ..utils.visualization import denormalize
from .checkpoint import CheckpointManager, load_checkpoint, resolve_resume_path
from .ema import ExponentialMovingAverage
from .losses import BetaScheduler

logger = logging.getLogger(__name__)

__all__ = ["Trainer", "build_optimizer", "build_lr_scheduler"]


def build_optimizer(config: Config, model: TextVAE) -> Optimizer:
    """AdamW/Adam/SGD over ``φ`` and ``θ`` (the prior is excluded by construction)."""
    optim_cfg = config.optim
    groups = model.parameter_groups(
        encoder_lr=optim_cfg.encoder_lr, decoder_lr=optim_cfg.decoder_lr
    )
    if not groups:
        raise ValueError("the model has no trainable parameters")
    name = optim_cfg.name.lower()
    betas = (float(optim_cfg.betas[0]), float(optim_cfg.betas[1]))
    if name == "adamw":
        return torch.optim.AdamW(
            groups,
            lr=optim_cfg.lr,
            betas=betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(groups, lr=optim_cfg.lr, betas=betas, eps=optim_cfg.eps)
    if name == "sgd":
        return torch.optim.SGD(
            groups, lr=optim_cfg.lr, momentum=0.9, weight_decay=optim_cfg.weight_decay
        )
    raise ValueError(f"unknown optimizer {optim_cfg.name!r}")


def build_lr_scheduler(
    config: Config, optimizer: Optimizer
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine/linear/constant decay down to ``min_lr_ratio × lr``."""
    optim_cfg = config.optim
    warmup = max(0, int(optim_cfg.warmup_steps))
    total = max(1, int(config.train.max_steps))
    floor = float(optim_cfg.min_lr_ratio)
    kind = optim_cfg.scheduler.lower()
    if kind not in {"cosine", "linear", "constant"}:
        raise ValueError(f"unknown lr scheduler {optim_cfg.scheduler!r}")

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
        decay = (
            0.5 * (1.0 + math.cos(math.pi * progress))
            if kind == "cosine"
            else 1.0 - progress
        )
        return floor + (1.0 - floor) * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Owns the optimisation loop, the schedules, logging and checkpointing."""

    def __init__(
        self,
        config: Config,
        model: TextVAE,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        accelerator: Optional[Any] = None,
    ) -> None:
        from accelerate import Accelerator

        self.config = config
        self.accelerator = accelerator or Accelerator(
            mixed_precision=config.train.mixed_precision,
            gradient_accumulation_steps=config.optim.grad_accum_steps,
        )
        if config.train.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.optimizer = build_optimizer(config, model)
        self.lr_scheduler = build_lr_scheduler(config, self.optimizer)

        # τ (§3.2) lives on the model so that inference paths see the same schedule.
        self.temperature_scheduler = model.temperature_scheduler
        self.beta_scheduler = BetaScheduler.from_config(config.loss)

        prepared = self.accelerator.prepare(
            model, self.optimizer, train_loader, self.lr_scheduler
        )
        self.model, self.optimizer, self.train_loader, self.lr_scheduler = prepared
        self.val_loader = (
            self.accelerator.prepare(val_loader) if val_loader is not None else None
        )

        # Built *after* prepare() so the shadow weights live on the training device.
        self.ema: Optional[ExponentialMovingAverage] = None
        if config.train.ema.enabled:
            self.ema = ExponentialMovingAverage(
                self.unwrapped,
                decay=config.train.ema.decay,
                warmup_steps=config.train.ema.warmup_steps,
            )

        self.output_dir = Path(config.train.output_dir) / config.train.run_name
        self.checkpoint_manager = CheckpointManager(
            self.output_dir / "checkpoints", keep_last_n=config.train.keep_last_n
        )
        self.metric_logger = MetricLogger(
            log_dir=self.output_dir / "logs",
            use_tensorboard=config.train.tracker.tensorboard,
            use_wandb=config.train.tracker.wandb,
            wandb_project=config.train.tracker.wandb_project,
            wandb_entity=config.train.tracker.wandb_entity,
            run_name=config.train.run_name,
            config=config.to_dict(),
            is_main_process=self.accelerator.is_main_process,
        )

        self.global_step = 0
        self.epoch = 0
        self._resume_if_requested()

    # ------------------------------------------------------------------ properties
    @property
    def unwrapped(self) -> TextVAE:
        """The underlying :class:`TextVAE` (unwrapped from DDP/AMP wrappers)."""
        return self.accelerator.unwrap_model(self.model)

    @property
    def is_main(self) -> bool:
        return bool(self.accelerator.is_main_process)

    @property
    def base_learning_rates(self) -> List[float]:
        """Peak LR per param group, read through accelerate's scheduler wrapper."""
        scheduler = getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)
        return list(getattr(scheduler, "base_lrs", []))

    # --------------------------------------------------------------------- resuming
    def _resume_if_requested(self) -> None:
        path = resolve_resume_path(
            self.config.train.resume, self.checkpoint_manager.directory
        )
        if path is None:
            return
        state = load_checkpoint(
            path,
            model=self.unwrapped,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            temperature_scheduler=self.temperature_scheduler,
            beta_scheduler=self.beta_scheduler,
            ema=self.ema,
            map_location="cpu",
            strict=False,
        )
        self.global_step = state.global_step
        self.epoch = state.epoch
        self._reapply_configured_learning_rates()
        logger.info(
            "Resumed from %s at step %d (epoch %d).", path, self.global_step, self.epoch
        )

    def _reapply_configured_learning_rates(self) -> None:
        """Let the config win over the checkpoint's learning rates.

        ``optimizer.load_state_dict`` restores each param group's ``lr`` and the LR
        scheduler restores its ``base_lrs``, so resuming with ``--set optim.lr=...`` would
        otherwise silently keep the old rate — the same trap as the β/τ schedules, which
        take their shape from the config on resume.
        """
        optim_cfg = self.config.optim
        overrides = {"encoder": optim_cfg.encoder_lr, "decoder": optim_cfg.decoder_lr}
        configured = [
            float(overrides.get(group.get("name", ""), None) or optim_cfg.lr)
            for group in self.optimizer.param_groups
        ]
        # Accelerate wraps the scheduler; assigning on the wrapper would only shadow the
        # attribute and leave the real LambdaLR untouched.
        scheduler = getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)
        restored = list(getattr(scheduler, "base_lrs", []))
        if restored and any(
            abs(new - old) > 1e-12 for new, old in zip(configured, restored)
        ):
            logger.info(
                "Learning rate differs from the checkpoint; keeping the configured value "
                "(checkpoint %s -> config %s).",
                restored,
                configured,
            )
        scheduler.base_lrs = configured
        for group, lr in zip(self.optimizer.param_groups, configured):
            group["lr"] = lr
        # Re-derive the current LR from the (possibly new) schedule at this step, bypassing
        # accelerate's wrapper so the update is not deferred to the next sync point.
        scheduler.step(self.global_step)

    # ------------------------------------------------------------------- train loop
    def fit(self) -> None:
        """Train until ``train.max_steps`` (or ``train.max_epochs``) is reached."""
        max_steps = self.config.train.max_steps
        max_epochs = self.config.train.max_epochs or math.inf
        logger.info(
            "Starting training: %d step(s), batch %d x %d accum x %d process(es).",
            max_steps,
            self.config.data.batch_size,
            self.config.optim.grad_accum_steps,
            self.accelerator.num_processes,
        )
        start = time.time()
        while self.global_step < max_steps and self.epoch < max_epochs:
            self._train_epoch(max_steps)
            self.epoch += 1
        # Final validation, samples and checkpoint.
        if self.val_loader is not None:
            self.validate()
        self.save_checkpoint()
        self.metric_logger.flush()
        self.metric_logger.close()
        logger.info(
            "Training finished: %d steps in %.1f min.",
            self.global_step,
            (time.time() - start) / 60,
        )

    def _train_epoch(self, max_steps: int) -> None:
        self.model.train()
        for batch in self.train_loader:
            if self.global_step >= max_steps:
                return
            metrics = self.train_step(batch)
            if metrics is None:  # gradient accumulation in progress
                continue
            self._after_step(metrics)

    def train_step(self, batch: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """One (possibly accumulated) optimisation step; ``None`` until the step lands."""
        images = batch["images"]
        caption_ids = batch.get("input_ids")

        with self.accelerator.accumulate(self.model):
            output: TextVAEOutput = self.model(
                images,
                temperature=self.temperature_scheduler.value(self.global_step),
                beta=self.beta_scheduler.value(self.global_step),
                caption_input_ids=caption_ids,
            )
            self.accelerator.backward(output.loss)
            # Read the flag *inside* the accumulate context: it tells us whether this
            # micro-batch closes an accumulation cycle (i.e. whether a real step lands).
            synced = bool(self.accelerator.sync_gradients)
            grad_norm = None
            if synced and self.config.optim.grad_clip > 0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config.optim.grad_clip
                )
            # Accelerate's wrappers make these no-ops while gradients are accumulating.
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        if not synced:
            return None

        self.global_step += 1
        self.temperature_scheduler.step(self.global_step)
        self.beta_scheduler.step(self.global_step)
        if self.ema is not None:
            self.ema.update(self.unwrapped)

        metrics = {
            "loss": float(output.loss.detach()),
            "diffusion_loss": float(output.diffusion_loss.detach()),
            "lr": float(self.lr_scheduler.get_last_lr()[0]),
            **output.metrics,
        }
        if grad_norm is not None:
            metrics["grad_norm"] = float(grad_norm)
        for name, value in output.aux_losses.items():
            metrics[name] = float(value.detach())
        self._last_output = output
        return metrics

    def _after_step(self, metrics: Dict[str, float]) -> None:
        train_cfg = self.config.train
        if train_cfg.log_every and self.global_step % train_cfg.log_every == 0:
            self.metric_logger.log_scalars(metrics, self.global_step, prefix="train/")
            self.metric_logger.log_console(
                {
                    k: metrics[k]
                    for k in (
                        "loss",
                        "diffusion_loss",
                        "kl",
                        "temperature",
                        "beta",
                        "lr",
                    )
                    if k in metrics
                },
                self.global_step,
            )
            if train_cfg.tracker.log_latent_text and hasattr(self, "_last_output"):
                texts = self.unwrapped.latent_texts(self._last_output.latent)
                self.metric_logger.log_text(
                    "train/latent_text",
                    texts[: train_cfg.num_log_samples],
                    self.global_step,
                )
        if train_cfg.val_every and self.global_step % train_cfg.val_every == 0:
            self.validate()
        if train_cfg.sample_every and self.global_step % train_cfg.sample_every == 0:
            self.log_samples()
        if train_cfg.ckpt_every and self.global_step % train_cfg.ckpt_every == 0:
            self.save_checkpoint()

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """Mean ELBO terms over (a prefix of) the validation loader."""
        if self.val_loader is None:
            return {}
        limit = max_batches or self.config.train.val_max_batches
        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        with self._ema_weights():
            for i, batch in enumerate(self.val_loader):
                if limit and i >= limit:
                    break
                output = self.model(
                    batch["images"],
                    temperature=self.config.model.gumbel.eval_temperature,
                    beta=self.beta_scheduler.value(self.global_step),
                    stochastic=True,
                )
                metrics = {
                    "loss": float(output.loss),
                    "diffusion_loss": float(output.diffusion_loss),
                    **output.metrics,
                }
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                count += 1
        self.model.train()
        if not count:
            return {}
        averaged = {key: value / count for key, value in totals.items()}
        self.metric_logger.log_scalars(averaged, self.global_step, prefix="val/")
        self.metric_logger.log_console(
            {k: averaged[k] for k in ("loss", "diffusion_loss", "kl") if k in averaged},
            self.global_step,
            prefix="[val] ",
        )
        self._last_val_loss = averaged.get("loss")
        return averaged

    @torch.no_grad()
    def log_samples(self, num_samples: Optional[int] = None) -> None:
        """Log image → text → image reconstructions (the interpretability artefact)."""
        if not self.is_main or not self.config.train.tracker.log_images:
            return
        loader: Optional[Iterable[Dict[str, Any]]] = (
            self.val_loader or self.train_loader
        )
        if loader is None:
            return
        batch = next(iter(loader))
        count = num_samples or self.config.train.num_log_samples
        images = batch["images"][:count]
        model = self.unwrapped
        was_training = model.training
        model.eval()
        with self._ema_weights():
            recon, texts, _ = model.reconstruct(images)
        if was_training:
            model.train()
        self.metric_logger.log_reconstruction_panel(
            "samples",
            denormalize(images),
            denormalize(recon),
            texts,
            self.global_step,
        )

    def _ema_weights(self) -> Any:
        """Context manager that swaps in the EMA weights when configured."""
        if self.ema is not None and self.config.train.ema.use_for_eval:
            return self.ema.applied_to(self.unwrapped)

        class _NoSwap:
            def __enter__(self_inner) -> None:
                return None

            def __exit__(self_inner, *exc: Any) -> None:
                return None

        return _NoSwap()

    # ---------------------------------------------------------------- checkpointing
    def save_checkpoint(self, metric: Optional[float] = None) -> Optional[Path]:
        """Persist the complete training state (main process only)."""
        self.accelerator.wait_for_everyone()
        if not self.is_main:
            return None
        return self.checkpoint_manager.save(
            global_step=self.global_step,
            metric=(
                metric if metric is not None else getattr(self, "_last_val_loss", None)
            ),
            model=self.unwrapped,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            temperature_scheduler=self.temperature_scheduler,
            beta_scheduler=self.beta_scheduler,
            ema=self.ema,
            epoch=self.epoch,
            config=self.config,
        )

    def state_summary(self) -> Dict[str, Any]:
        """Small dict describing the run (used in logs and tests)."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "temperature": self.temperature_scheduler.value(self.global_step),
            "beta": self.beta_scheduler.value(self.global_step),
            "trainable_parameters": self.unwrapped.num_trainable_parameters(),
        }
