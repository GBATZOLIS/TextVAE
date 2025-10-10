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
        self.recon_criterion = nn.MSELoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0
        # --- NEW: List to store sequence lengths for histogram logging ---
        seq_lengths_epoch = []

        progress_bar = tqdm(
            self.train_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Training]"
        )
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(self.device)
            self.optimizer.zero_grad()

            seq_len = None
            if self.config.APPLY_TRUNCATION:
                n = self.config.NUM_PATCHES
                min_len = int(n * self.config.MIN_SEQ_LEN_FRAC)
                seq_len = random.randint(min_len, n)
                # Store the length for the epoch-end histogram
                seq_lengths_epoch.append(seq_len)

            model_output = self.model(images, seq_len=seq_len)
            reconstructed_images, vq_loss, perplexity = (
                model_output["reconstructions"],
                model_output["vq_loss"],
                model_output["perplexity"],
            )

            recon_loss = self.recon_criterion(reconstructed_images, images)
            total_loss_batch = recon_loss + vq_loss

            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()

            if batch_idx % self.config.LOG_INTERVAL == 0:
                # Log batch-level metrics
                wandb.log(
                    {
                        "train/batch_loss": total_loss_batch.item(),
                        "train/recon_loss": recon_loss.item(),
                        "train/vq_loss": vq_loss.item(),
                        "train/perplexity": perplexity.item(),
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
        # --- UPDATED: Log epoch-level metrics and the histogram ---
        epoch_logs = {
            "train/avg_epoch_loss": total_loss / num_batches,
            "train/avg_epoch_recon_loss": total_recon_loss / num_batches,
            "train/avg_epoch_vq_loss": total_vq_loss / num_batches,
            "train/avg_epoch_perplexity": total_perplexity / num_batches,
            "epoch": epoch,
        }
        if self.config.APPLY_TRUNCATION and seq_lengths_epoch:
            epoch_logs["train/seq_len_histogram"] = wandb.Histogram(seq_lengths_epoch)
        wandb.log(epoch_logs)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0

        progress_bar = tqdm(
            self.val_loader, desc=f"Epoch {epoch}/{self.config.NUM_EPOCHS} [Validation]"
        )
        for images, _ in progress_bar:
            images = images.to(self.device)

            model_output = self.model(images, seq_len=None)
            reconstructed_images, vq_loss, perplexity = (
                model_output["reconstructions"],
                model_output["vq_loss"],
                model_output["perplexity"],
            )

            recon_loss = self.recon_criterion(reconstructed_images, images)
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

            model_dir = os.path.join(
                wandb.run.dir, f"patch_{self.config.PATCH_SIZE}/checkpoints"
            )
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
            torch.save(self.model.state_dict(), model_path)
