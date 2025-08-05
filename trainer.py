# trainer.py

"""
Contains the training and validation logic for the Text-Conditioned VAE.
This version includes robust KL annealing and gradient clipping to prevent
mode collapse and stabilize training.
"""
import logging
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from config import VAEConfig
from models.vae import TextVAE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAETrainer:
    """Manages the training loops, checkpointing, and recovery."""

    def __init__(self, cfg: VAEConfig, work_dir: str):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = TextVAE(cfg).to(self.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(
            f"Number of trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M"
        )
        self.optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
        self.scaler = GradScaler(device=self.device_type, enabled=cfg.use_amp)

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # --- KL Annealing Schedule Parameters ---
        # These can be tuned in your config file if needed
        self.kl_anneal_warmup_steps = 5000  # Steps with kl_weight = 0
        self.kl_anneal_total_steps = 25000  # Total steps to reach full kl_weight

    def get_current_kl_weight(self) -> float:
        """Calculates the KL weight for the current step based on the annealing schedule."""
        if self.global_step < self.kl_anneal_warmup_steps:
            return 0.0

        # Linearly increase the weight from 0 to the target cfg.kl_weight
        progress = (self.global_step - self.kl_anneal_warmup_steps) / (
            self.kl_anneal_total_steps - self.kl_anneal_warmup_steps
        )
        current_weight = min(1.0, progress) * self.cfg.kl_weight
        return current_weight

    def load_checkpoint(self, path: str):
        """Loads a checkpoint to resume training."""
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )  # Allow loading config
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint["global_step"]
            self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            logger.info(
                f"Resumed training from epoch {self.epoch} at step {self.global_step}."
            )
        except Exception as e:
            logger.error(
                f"Could not load checkpoint due to error: {e}. Starting from scratch."
            )

    def save_checkpoint(self, filename: str):
        """Saves a model checkpoint."""
        checkpoint_path = self.work_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> float:
        """Runs a single epoch and returns the average loss."""
        self.model.train(is_training)
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.epoch+1}/{self.cfg.epochs} [{'Train' if is_training else 'Valid'}]",
        )
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for batch in progress_bar:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device)

            with torch.set_grad_enabled(is_training):
                with autocast(device_type=self.device_type, enabled=self.cfg.use_amp):
                    losses = self.model(images)

                    # FIX: Apply the KL annealing schedule during training
                    current_kl_weight = (
                        self.get_current_kl_weight()
                        if is_training
                        else self.cfg.kl_weight
                    )
                    loss = losses["recon"] + current_kl_weight * losses["kl"]

            if is_training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                # FIX: Add gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1

                if (
                    self.global_step > 0
                    and self.global_step % self.cfg.save_every_n_steps == 0
                ):
                    self.save_checkpoint(filename=f"step_{self.global_step}.pt")

            total_loss += loss.item()  # Log the combined loss
            total_recon += losses["recon"].item()
            total_kl += losses["kl"].item()
            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Recon": f"{losses['recon'].item():.4f}",
                    "KL": f"{losses['kl'].item():.4f}",
                    "KL_W": f"{current_kl_weight:.6f}",
                }
            )

        avg_loss = total_loss / len(dataloader)
        logger.info(
            f"Epoch {self.epoch+1} Avg {'Train' if is_training else 'Valid'} Loss: {avg_loss:.4f}"
        )
        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """The main training loop."""
        logger.info("--- Starting End-to-End Training with Swin Transformer ---")
        if self.epoch == 0:
            self.save_checkpoint(filename="initial_model.pt")

        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            self._run_epoch(train_loader, is_training=True)

            with torch.no_grad():
                val_loss = self._run_epoch(val_loader, is_training=False)

            self.save_checkpoint(filename=f"epoch_{self.epoch+1}.pt")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(
                    f"New best validation loss: {self.best_val_loss:.4f}. Saving best model."
                )
                self.save_checkpoint(filename="best_model.pt")

        logger.info(f"--- Training Finished after {self.cfg.epochs} epochs ---")
