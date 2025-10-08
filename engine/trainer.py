# engine/trainer.py

import torch
import torch.nn as nn
import random
from tqdm import tqdm
import wandb
from utils.visualize import save_reconstruction_sample
import os


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.pixel_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Training]"
        )
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(self.device)

            pixels_quantized = ((images * 0.5 + 0.5) * 255).long()
            pixels_flat = pixels_quantized.permute(0, 2, 3, 1).reshape(
                images.size(0), -1
            )

            pixel_input, pixel_target = pixels_flat[:, :-1], pixels_flat[:, 1:]

            self.optimizer.zero_grad()

            current_n_generate = random.randint(1, self.config.MAX_N_GENERATE)

            model_output = self.model(images, pixel_input, current_n_generate)
            logits, vq_loss, perplexity = (
                model_output["pixel_logits"],
                model_output["vq_loss"],
                model_output["perplexity"],
            )

            recon_loss = self.pixel_criterion(
                logits.reshape(-1, logits.size(-1)), pixel_target.reshape(-1)
            )
            total_loss_batch = recon_loss + vq_loss

            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

            if batch_idx % self.config.LOG_INTERVAL == 0:
                wandb.log(
                    {
                        "train/batch_loss": total_loss_batch.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/vq_loss": vq_loss.item(),
                        "train/perplexity": perplexity.item(),
                        "train/N_generated": current_n_generate,
                        "epoch": epoch,
                    }
                )

            progress_bar.set_postfix(
                {
                    "Loss": f"{total_loss_batch.item():.4f}",
                    "Perplexity": f"{perplexity.item():.2f}",
                }
            )

        num_batches = len(self.train_loader)
        wandb.log(
            {
                "train/avg_epoch_loss": total_loss / num_batches,
                "train/avg_epoch_recon_loss": total_recon_loss / num_batches,
                "train/avg_epoch_vq_loss": total_vq_loss / num_batches,
                "train/avg_epoch_perplexity": total_perplexity / num_batches,
                "epoch": epoch,
            }
        )

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0
        n_generate = self.config.MAX_N_GENERATE // 2

        progress_bar = tqdm(
            self.val_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Validation]"
        )
        for images, _ in progress_bar:
            images = images.to(self.device)

            pixels_quantized = ((images * 0.5 + 0.5) * 255).long()
            pixels_flat = pixels_quantized.permute(0, 2, 3, 1).reshape(
                images.size(0), -1
            )
            pixel_input, pixel_target = pixels_flat[:, :-1], pixels_flat[:, 1:]

            model_output = self.model(images, pixel_input, n_generate)
            logits, vq_loss, perplexity = (
                model_output["pixel_logits"],
                model_output["vq_loss"],
                model_output["perplexity"],
            )

            recon_loss = self.pixel_criterion(
                logits.reshape(-1, logits.size(-1)), pixel_target.reshape(-1)
            )
            total_loss_batch = recon_loss + vq_loss

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

            progress_bar.set_postfix(
                {
                    "Val Loss": f"{total_loss_batch.item():.4f}",
                    "Val Perplexity": f"{perplexity.item():.2f}",
                }
            )

        num_batches = len(self.val_loader)
        wandb.log(
            {
                "val/avg_epoch_loss": total_loss / num_batches,
                "val/avg_epoch_recon_loss": total_recon_loss / num_batches,
                "val/avg_epoch_vq_loss": total_vq_loss / num_batches,
                "val/avg_epoch_perplexity": total_perplexity / num_batches,
                "epoch": epoch,
            }
        )
        print(
            f"Validation Epoch {epoch} Avg Loss: {total_loss / num_batches:.4f}, Avg Perplexity: {total_perplexity / num_batches:.2f}"
        )

    def run(self):
        train_sample_batch = next(iter(self.train_loader))[0].to(self.device)
        val_sample_batch = next(iter(self.val_loader))[0].to(self.device)

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

            if epoch % self.config.SAVE_IMAGE_INTERVAL == 0:
                save_reconstruction_sample(
                    self.model,
                    train_sample_batch,
                    epoch,
                    self.device,
                    self.config,
                    prefix="train",
                )
                save_reconstruction_sample(
                    self.model,
                    val_sample_batch,
                    epoch,
                    self.device,
                    self.config,
                    prefix="validation",
                )

            model_path = os.path.join(wandb.run.dir, f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), model_path)
