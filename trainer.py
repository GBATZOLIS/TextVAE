# trainer.py
import bitsandbytes as bnb
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
import os

from config import AppConfig
from models.codec import Codec

logger = logging.getLogger(__name__)


class CodecTrainer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=None,  # config.system.mixed_precision,
            log_with="tensorboard",
            project_dir=str(config.system.output_dir),
        )
        logger.info(f"Using device: {self.accelerator.device}")
        self.model = Codec(config, self.accelerator.device)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.adam_weight_decay,
        )
        self.log_parameter_status()

    def log_parameter_status(self):
        logger.info("--- Codec Parameter Status ---")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info("The following parameters are trainable:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.info(f"  - {name:<70} | Shape: {str(list(param.shape)):<20}")
        logger.info("-" * 100)
        logger.info(f"Total Model Parameters: {total_params / 1e6:.2f}M")
        logger.info(f"Trainable Parameters:   {trainable_params_count / 1e6:.2f}M")
        if total_params > 0:
            logger.info(
                f"Fine-tuning ratio:      {100 * trainable_params_count / total_params:.4f}%"
            )
        logger.info("--- End of Status ---")

    def fit(self, train_loader):
        self.model, self.optimizer, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader
        )
        global_step = 0
        logger.info("--- Starting Training ---")
        for epoch in range(self.config.training.num_train_epochs):
            self.model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                images_tensor = batch[0] if isinstance(batch, (list, tuple)) else batch
                # with self.accelerator.autocast():
                outputs = self.model(images_tensor)
                loss = (
                    outputs["loss"] / self.config.training.gradient_accumulation_steps
                )

                self.accelerator.backward(loss)
                if (
                    global_step + 1
                ) % self.config.training.gradient_accumulation_steps == 0:
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                effective_step = (
                    global_step // self.config.training.gradient_accumulation_steps
                )
                if (
                    effective_step > 0
                    and effective_step % self.config.training.checkpointing_steps == 0
                ):
                    output_dir = os.path.join(
                        self.config.system.output_dir, f"checkpoint-{effective_step}"
                    )
                    self.accelerator.save_state(output_dir)
                    logger.info(f"Saved checkpoint to {output_dir}")

                global_step += 1
