# trainer.py

"""
Contains the training and validation logic for the Text-Conditioned VAE.
Separating the trainer from the model definition keeps the code organized.
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

# Set up a logger for this module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class TextVAETrainer:
    """
    Manages the training and validation loops for the TextVAE model.
    Includes periodic checkpointing and saving the best model.
    """

    def __init__(self, cfg: VAEConfig, work_dir: str = "./runs"):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        # --- Device and Model Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.model = TextVAE(cfg).to(self.device)

        # --- Optimizer and Scaler ---
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
        self.scaler = GradScaler(device=self.device_type, enabled=cfg.use_amp)

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")  # Track the best validation loss

    def _run_epoch(self, dataloader: DataLoader, is_training: bool) -> float:
        """
        Helper function to run a single epoch of training or validation.
        Returns the average loss for the epoch.
        """
        self.model.train(is_training)

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.epoch+1}/{self.cfg.epochs} [{'Train' if is_training else 'Valid'}]",
        )

        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for batch in progress_bar:
            # For simplicity, assuming batch is just images. If it's a dict/tuple, adjust accordingly.
            images = batch.to(self.device)

            with torch.set_grad_enabled(is_training):
                with autocast(device_type=self.device_type, enabled=self.cfg.use_amp):
                    losses = self.model(images)
                    loss = losses["loss"]

            if is_training:
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1

                # --- New: Periodic Checkpointing ---
                if (
                    self.global_step > 0
                    and self.global_step % self.cfg.save_every_n_steps == 0
                ):
                    self.save_checkpoint(is_step_checkpoint=True)

            # --- Logging and Metrics ---
            total_loss += losses["loss"].item()
            total_recon += losses["recon"].item()
            total_kl += losses["kl"].item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{losses['loss'].item():.4f}",
                    "Recon": f"{losses['recon'].item():.4f}",
                    "KL": f"{losses['kl'].item():.4f}",
                }
            )

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        logger.info(
            f"Epoch {self.epoch+1} {'Train' if is_training else 'Valid'} Avg Loss: {avg_loss:.4f}, "
            f"Avg Recon: {avg_recon:.4f}, Avg KL: {avg_kl:.4f}"
        )
        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """The main training loop."""
        logger.info("--- Starting Training ---")
        # Save an initial checkpoint before training starts
        self.save_checkpoint(filename="initial_model.pt")

        for epoch in range(self.cfg.epochs):
            self.epoch = epoch

            # Training phase
            self._run_epoch(train_loader, is_training=True)

            # Validation phase
            val_loss = self._run_epoch(val_loader, is_training=False)

            # --- Save Checkpoint at end of epoch ---
            self.save_checkpoint(is_epoch_checkpoint=True)

            # --- New: Save the best model based on validation loss ---
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(
                    f"New best validation loss: {self.best_val_loss:.4f}. Saving best model."
                )
                self.save_checkpoint(filename="best_model.pt")

        logger.info("--- Training Finished ---")

    def save_checkpoint(
        self,
        is_epoch_checkpoint: bool = False,
        is_step_checkpoint: bool = False,
        filename: str = "./runs/checkpoint.pt",
    ):
        """
        Saves a model checkpoint.
        - `is_epoch_checkpoint`: If True, saves as epoch_{epoch_num}.pt
        - `is_step_checkpoint`: If True, saves as step_{step_num}.pt
        - `filename`: If provided, uses this exact filename (e.g., 'best_model.pt').
        """
        if filename:
            assert isinstance(
                filename, str
            ), f"expeted filename to be str but got  {type(filename)} instead"
            checkpoint_path = self.work_dir / filename
        elif is_step_checkpoint:
            checkpoint_path = self.work_dir / f"step_{self.global_step}.pt"
        elif is_epoch_checkpoint:
            checkpoint_path = self.work_dir / f"epoch_{self.epoch+1}.pt"
        else:
            # Fallback for any other case
            checkpoint_path = self.work_dir / "latest.pt"

        # Save only the trainable parts of the model
        trainable_state_dict = {
            "vision_enc": self.model.vision_enc.state_dict(),
            "text_dec": self.model.text_dec.state_dict(),
            "vision_feat_proj": self.model.vision_feat_proj.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(trainable_state_dict, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
