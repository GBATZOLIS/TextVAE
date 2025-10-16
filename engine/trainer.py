# engine/trainer.py

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import wandb
import torchvision.utils as vutils
import os
from .vgg_loss import VGGPerceptualLoss
from models.prior import CodebookPrior


class VQVAETrainer:
    # --- UPDATE: Accept the scheduler in the constructor ---
    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, device, config
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler  # Store the scheduler
        self.device = device
        self.config = config
        self.recon_criterion = nn.MSELoss()
        self.perceptual_loss = VGGPerceptualLoss().to(device)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, total_recon_loss, total_vq_loss, total_perplexity = 0, 0, 0, 0
        total_mse_loss, total_perceptual_loss = 0, 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Training VQ-VAE]")

        for images, _ in progress_bar:
            images = images.to(self.device)
            self.optimizer.zero_grad()

            seq_len_tensor = None
            if self.config.APPLY_TRUNCATION:
                n = self.config.NUM_PATCHES
                b = images.shape[0]
                min_len = n // 2
                seq_len_tensor = torch.randint(min_len, n + 1, (b,), device=self.device)

            model_output = self.model(images, seq_len=seq_len_tensor)
            reconstructed_images = model_output["reconstructions"]
            vq_loss = model_output["vq_loss"]
            perplexity = model_output["perplexity"]

            mse_loss = self.recon_criterion(reconstructed_images, images)
            p_loss = self.perceptual_loss(reconstructed_images, images)
            recon_loss = mse_loss + self.config.PERCEPTUAL_LOSS_WEIGHT * p_loss

            total_loss_batch = recon_loss + vq_loss
            total_loss_batch.backward()
            self.optimizer.step()

            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            total_perplexity += perplexity.item()
            total_mse_loss += mse_loss.item()
            total_perceptual_loss += p_loss.item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{total_loss_batch.item():.4f}",
                    "Perplexity": f"{perplexity.item():.2f}",
                }
            )

        num_batches = len(self.train_loader)

        # --- UPDATE: Step the scheduler after each epoch ---
        self.scheduler.step()

        epoch_logs = {
            "train/vqvae_avg_epoch_loss": total_loss / num_batches,
            "train/vqvae_avg_epoch_recon_loss": total_recon_loss / num_batches,
            "train/vqvae_avg_epoch_mse_loss": total_mse_loss / num_batches,
            "train/vqvae_avg_epoch_perceptual_loss": total_perceptual_loss
            / num_batches,
            "train/vqvae_avg_epoch_vq_loss": total_vq_loss / num_batches,
            "train/vqvae_avg_epoch_perplexity": total_perplexity / num_batches,
            "train/learning_rate": self.scheduler.get_last_lr()[
                0
            ],  # Log the current LR
            "epoch": epoch,
        }
        wandb.log(epoch_logs)

    @torch.no_grad()
    def save_reconstructions(self, epoch, val_images):
        self.model.eval()

        full_recon = self.model(val_images, seq_len=None)["reconstructions"]

        n = self.config.NUM_PATCHES
        b = val_images.shape[0]
        min_len = n // 4
        seq_len_tensor = torch.randint(min_len, n, (b,), device=self.device)
        trunc_recon = self.model(val_images, seq_len=seq_len_tensor)["reconstructions"]

        val_images = val_images.mul(0.5).add(0.5)
        full_recon = full_recon.mul(0.5).add(0.5)
        trunc_recon = trunc_recon.mul(0.5).add(0.5)

        comparison = torch.cat([val_images, full_recon, trunc_recon])

        save_dir = "outputs/reconstructions"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"reconstruction_epoch_{epoch}.png")

        vutils.save_image(comparison.clamp(0, 1), save_path, nrow=val_images.size(0))
        wandb.log(
            {
                "vqvae_reconstructions": wandb.Image(
                    save_path,
                    caption="Top: Original, Middle: Full Recon, Bottom: Truncated Recon",
                )
            }
        )

    @torch.no_grad()
    def sample_and_show_truncations(self, val_images):
        print("Generating samples from truncated sequences...")
        self.model.eval()

        patches = self.model.patch_embedding(val_images)
        encoded_features = self.model.encoder(patches)
        quantized_features, _, _, _ = self.model.quantizer(encoded_features)

        b, n, d = quantized_features.shape
        truncation_levels = [0.25, 0.50, 0.75, 1.0]
        num_images_to_show = 8

        all_images = [val_images[:num_images_to_show]]

        for level in truncation_levels:
            seq_len = int(n * level)
            if seq_len == 0:
                continue

            context = quantized_features[:num_images_to_show, :seq_len, :]

            reconstructions = self.model.decoder(context, key_padding_mask=None)
            all_images.append(reconstructions)

        comparison_grid = torch.cat(all_images)
        comparison_grid = comparison_grid.mul(0.5).add(0.5)
        save_dir = "outputs/samples"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "truncation_samples.png")
        vutils.save_image(
            comparison_grid.clamp(0, 1), save_path, nrow=num_images_to_show
        )
        wandb.log(
            {
                "truncation_samples": wandb.Image(
                    save_path,
                    caption="Top: Original. Rows 2-5: Reconstructions from 25%, 50%, 75%, 100% of sequence.",
                )
            }
        )
        print(f"Truncation samples saved to {save_path}")

    def train_prior(self):
        print("\n--- Starting Training for the Codebook Prior ---")
        self.model.eval()

        prior_model = CodebookPrior(
            num_codes=self.config.NUM_EMBEDTINGS,
            embedding_dim=self.config.PRIOR_EMBEDDING_DIM,
            nhead=self.config.PRIOR_NHEAD,
            num_layers=self.config.PRIOR_LAYERS,
            dropout=self.config.PRIOR_DROPOUT,
        ).to(self.device)

        prior_optimizer = torch.optim.Adam(
            prior_model.parameters(), lr=self.config.PRIOR_LEARNING_RATE
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.config.PRIOR_NUM_EPOCHS + 1):
            prior_model.train()
            total_loss = 0
            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch} [Training Prior]"
            )

            for images, _ in progress_bar:
                images = images.to(self.device)
                prior_optimizer.zero_grad()

                with torch.no_grad():
                    model_output = self.model(images)
                    targets = model_output["indices"]

                logits = prior_model(targets)

                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                loss.backward()
                prior_optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"Prior Loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)
            wandb.log({"prior/avg_epoch_loss": avg_loss, "prior_epoch": epoch})

        torch.save(prior_model.state_dict(), self.config.PRIOR_CHECKPOINT_PATH)
        print(f"Prior model saved to {self.config.PRIOR_CHECKPOINT_PATH}")
        return prior_model

    @torch.no_grad()
    def generate_with_prior(self, prior_model, num_samples=64):
        print("Generating new images with the trained prior...")
        prior_model.eval()

        sos_index = self.config.NUM_EMBEDTINGS
        start_tokens = torch.full(
            (num_samples, 1), sos_index, device=self.device, dtype=torch.long
        )

        generated_indices = start_tokens

        for _ in range(self.config.NUM_PATCHES):
            token_embeds = prior_model.token_embedding(generated_indices)
            seq = (
                token_embeds
                + prior_model.positional_embedding[:, : token_embeds.size(1), :]
            )

            mask = prior_model.generate_square_subsequent_mask(
                token_embeds.size(1), self.device
            )

            output = prior_model.transformer_decoder(
                seq, seq, tgt_mask=mask, memory_mask=mask
            )
            logits = prior_model.output_head(output[:, -1, :])

            top_k_logits, top_k_indices = torch.topk(logits, k=50, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_local_idx = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_local_idx)

            generated_indices = torch.cat([generated_indices, next_token], dim=1)

        final_indices = generated_indices[:, 1:]

        quantized = self.model.quantizer.embedding(final_indices).view(
            num_samples, self.config.NUM_PATCHES, -1
        )

        generated_images = self.model.decoder(quantized)
        generated_images = generated_images.mul(0.5).add(0.5)

        save_dir = "outputs/generated"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "prior_generated_samples.png")
        vutils.save_image(generated_images.clamp(0, 1), save_path, nrow=8)
        wandb.log({"generative_samples": wandb.Image(save_path)})
        print(f"Generative samples saved to {save_path}")

    def run(self):
        val_sample_batch = next(iter(self.val_loader))[0].to(self.device)

        # --- VQ-VAE Training Stage ---
        for epoch in range(1, self.config.VQVAE_NUM_EPOCHS + 1):
            self.train_epoch(epoch)
            if epoch % 10 == 0 or epoch == self.config.VQVAE_NUM_EPOCHS:
                self.save_reconstructions(epoch, val_sample_batch[:16])

        torch.save(self.model.state_dict(), self.config.VQVAE_CHECKPOINT_PATH)
        print(f"VQ-VAE model saved to {self.config.VQVAE_CHECKPOINT_PATH}")
        self.sample_and_show_truncations(val_sample_batch)

        # --- Prior Training Stage ---
        trained_prior = self.train_prior()

        # --- Generative Sampling Stage ---
        self.generate_with_prior(trained_prior)
