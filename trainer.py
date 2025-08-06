# trainer.py
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging

from config import AppConfig
from models.vae import TextVAE  # Import the single, consolidated VAE model

logger = logging.getLogger(__name__)


class TextVAETrainer:
    """Orchestrates the training of the consolidated TextVAE model."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.system.mixed_precision,
            log_with="tensorboard",
            project_dir=str(config.system.output_dir),
        )
        logger.info(f"Using device: {self.accelerator.device}")

        # --- Model Initialization ---
        # Instantiate the single TextVAE model. It handles all sub-module setup internally.
        self.model = TextVAE(config, self.accelerator.device)

        # --- Optimizer ---
        # The TextVAE internally freezes the necessary parts. We just collect the
        # parameters that are left as trainable.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,  # Pass only the trainable parameters
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.adam_weight_decay,
        )

        self.log_parameter_status()

    def log_parameter_status(self):
        """Logs the name, shape, and trainable status of all model parameters."""
        logger.info("--- TextVAE Parameter Status ---")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logger.info("The following parameters are trainable:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"  - {name:<70} | Shape: {str(list(param.shape)):<20}")

        logger.info("-" * 100)
        logger.info(f"Total Model Parameters: {total_params / 1e6:.2f}M")
        logger.info(f"Trainable Parameters:   {trainable_params / 1e6:.2f}M")
        logger.info(
            f"Fine-tuning ratio:      {100 * trainable_params / total_params:.4f}%"
        )
        logger.info("--- End of Status ---")

    def fit(self, train_loader, val_loader):
        # Prepare the model and optimizer with the accelerator
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )

        logger.info("--- Starting Training ---")
        for epoch in range(self.config.training.num_train_epochs):
            self.model.train()  # Set the model to training mode
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}",
                disable=not self.accelerator.is_local_main_process,
            )

            for batch in progress_bar:
                # The dataset now returns only the image tensor
                images_tensor, _ = batch

                # Use autocast for mixed precision
                with self.accelerator.autocast():
                    # The forward pass is now a single call to our model
                    outputs = self.model(images_tensor)
                    loss = outputs["loss"]

                # Standard optimization steps
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                # Update progress bar with latest loss values
                progress_bar.set_postfix(
                    Loss=f"{loss.item():.4f}",
                    Recon=f"{outputs['recon_loss'].item():.4f}",
                    KL=f"{outputs['kl_loss'].item():.4f}",
                )

        logger.info("--- Training Finished ---")
