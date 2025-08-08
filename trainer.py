# trainer.py
import logging
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ProjectConfig
from dataset import ImageDataset
from models.plausibility import PlausibilityModule

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the training process using the DeepSpeed engine.
    """

    def __init__(self, config: ProjectConfig, model_engine, train_dataloader):
        self.config = config
        self.model_engine = model_engine
        self.train_dataloader = train_dataloader

        self.best_val_loss = float("inf")
        self.global_step = self.model_engine.global_steps  # Use DeepSpeed's global step

        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.project_name,
                name=self.config.logging.run_name,
                config=vars(config),
            )
            wandb.watch(self.model_engine, log="all")

        self._setup_validation_loader()

        # The plausibility module is not trained, so we manage it separately
        self.plausibility_module = PlausibilityModule(
            model_name=self.config.models.plausibility_module_name
        ).to(self.model_engine.device)

    def _setup_validation_loader(self):
        """Initializes the validation dataloader."""
        # This re-uses the training dataset object but with a different split logic
        # For simplicity, we'll create a new dataset object for validation
        try:
            full_dataset = ImageDataset(
                root=self.config.data.data_dir, image_size=self.config.data.image_size
            )
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(42)
            _, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=generator
            )

            self.valid_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            logger.info(
                f"Validation Dataloader created with {len(val_dataset)} samples."
            )
        except FileNotFoundError:
            logger.warning(
                "Could not create validation loader, validation will be skipped."
            )
            self.valid_dataloader = None

    def _run_validation(self):
        """Performs a full validation run."""
        if self.valid_dataloader is None:
            return {"val_loss": -1.0}

        self.model_engine.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(self.valid_dataloader, desc="Validating", leave=False)
            for batch in pbar:
                original_images = batch.to(self.model_engine.device, dtype=torch.half)

                generated_logits, _, loss_recon = self.model_engine(original_images)
                loss_plausibility = self.plausibility_module(generated_logits)

                total_loss = (
                    loss_recon
                    + self.config.training.lambda_plausibility * loss_plausibility
                )
                total_val_loss += total_loss.item()

        avg_val_loss = total_val_loss / len(self.valid_dataloader)
        return {"val_loss": avg_val_loss}

    def save_checkpoint(self, is_best=False):
        """Saves a DeepSpeed checkpoint."""
        tag = f"step_{self.global_step}"
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        self.model_engine.save_checkpoint(checkpoint_dir, tag=tag)
        logger.info(
            f"Saved DeepSpeed checkpoint for step {self.global_step} to {checkpoint_dir}/{tag}"
        )
        if is_best:
            # You can add logic here to copy the best checkpoint if desired
            logger.info("New best model checkpoint saved.")

    def load_checkpoint(self, checkpoint_path):
        """Loads a DeepSpeed checkpoint."""
        _, client_sd = self.model_engine.load_checkpoint(checkpoint_path)
        self.global_step = client_sd.get("global_steps", 0)
        logger.info(
            f"Loaded checkpoint from {checkpoint_path} at global step {self.global_step}"
        )

    def train(self):
        """The main training loop, driven by the DeepSpeed engine."""
        logger.info("Starting training...")
        for epoch in range(self.config.training.epochs):
            self.model_engine.train()
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
            )
            for batch in pbar:
                original_images = batch.to(self.model_engine.device, dtype=torch.half)

                generated_logits, _, loss_recon = self.model_engine(original_images)
                loss_plausibility = self.plausibility_module(generated_logits.detach())
                total_loss = (
                    loss_recon
                    + self.config.training.lambda_plausibility * loss_plausibility
                )

                self.model_engine.backward(total_loss)
                self.model_engine.step()

                self.global_step = self.model_engine.global_steps
                log_data = {"train_loss": total_loss.item()}
                pbar.set_postfix(log_data)

                if self.config.logging.use_wandb:
                    wandb.log({**log_data, "global_step": self.global_step})

                # --- Step-based Checkpointing ---
                if (
                    self.global_step > 0
                    and self.global_step % self.config.logging.save_every_n_steps == 0
                ):
                    self.save_checkpoint()

            # --- End of Epoch Validation ---
            val_log = self._run_validation()
            is_best = val_log["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_log["val_loss"]

            logger.info(f"Epoch {epoch + 1} | Val Loss: {self.best_val_loss:.4f}")
            if self.config.logging.use_wandb:
                wandb.log({**val_log, "epoch": epoch})

            if (epoch + 1) % self.config.logging.save_every_n_epochs == 0:
                self.save_checkpoint(is_best=is_best)

        logger.info("Training finished.")
